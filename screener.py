# screener.py
# MASTER – Decision Screener (DK + US)
# Single source of truth: data/ticker_alias.csv

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "screener_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA_DIR / "ticker_alias.csv"
DIVCLASS_FILE = DATA_DIR / "dividend_classes.csv"
OUTPUT_FILE = RESULTS_DIR / "screener_results.csv"

# -------------------------
# LOAD TICKERS
# -------------------------
alias_df = pd.read_csv(ALIAS_FILE)
alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()
tickers = alias_df["Ticker"].dropna().unique().tolist()

alias_map = alias_df.set_index("Ticker").to_dict("index")

# Dividend class (King / Aristocrat)
divclass_map = {}
if DIVCLASS_FILE.exists():
    dc = pd.read_csv(DIVCLASS_FILE)
    dc["Ticker"] = dc["Ticker"].str.upper()
    divclass_map = dict(zip(dc["Ticker"], dc["Class"]))

print(f"🔥 Universe size: {len(tickers)} tickers")

rows = []

# -------------------------
# HELPERS
# -------------------------
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except:
        return None

def calc_score(yield_pct, cagr, payout, pe):
    score = 0
    if yield_pct:
        score += min(yield_pct * 5, 25)
    if cagr:
        score += min(cagr * 3, 30)
    if payout:
        if payout < 70:
            score += 20
        elif payout < 90:
            score += 10
    if pe:
        if pe < 18:
            score += 15
        elif pe < 25:
            score += 5
    return round(score, 0)

def signal_from_score(score):
    if score >= 80:
        return "GOLD"
    if score >= 65:
        return "BUY"
    if score >= 50:
        return "HOLD"
    return "WATCH"

def confidence_from_metrics(cagr, payout):
    if cagr and cagr >= 8 and payout and payout < 75:
        return "High"
    if cagr and cagr >= 4:
        return "Medium"
    return "Low"

# -------------------------
# MAIN LOOP
# -------------------------
for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        yield_pct = safe(info.get("dividendYield"))
        payout = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        if yield_pct:
            yield_pct *= 100
        if payout:
            payout *= 100

        # Dividend CAGR (5Y)
        divs = t.dividends
        cagr = None
        if divs is not None and len(divs) >= 6:
            try:
                start = divs.iloc[-6]
                end = divs.iloc[-1]
                if start > 0:
                    cagr = ((end / start) ** (1 / 5) - 1) * 100
            except:
                pass

        score = calc_score(yield_pct, cagr, payout, pe)
        signal = signal_from_score(score)
        conf = confidence_from_metrics(cagr, payout)

        why = []
        if ticker in divclass_map:
            why.append(divclass_map[ticker])
        if cagr and cagr >= 8:
            why.append("High CAGR")
        if payout and payout < 75:
            why.append("Safe payout")

        rows.append({
            "Ticker": ticker,
            "Name": info.get("longName", alias_map.get(ticker, {}).get("Name")),
            "Country": alias_map.get(ticker, {}).get("Country"),
            "Sector": alias_map.get(ticker, {}).get("Sector"),
            "Currency": info.get("currency"),
            "Price": price,
            "Yield %": round(yield_pct, 2) if yield_pct else None,
            "DivCAGR(5Y)": round(cagr, 2) if cagr else None,
            "Score": score,
            "Signal": signal,
            "Conf": conf,
            "Why": ", ".join(why) if why else None
        })

    except Exception as e:
        print(f"⚠ {ticker}: {e}")

# -------------------------
# SAVE
# -------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")
