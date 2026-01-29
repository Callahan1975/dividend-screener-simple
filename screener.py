import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import math

# ----------------------------
# CONFIG
# ----------------------------

TICKERS = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "TROW", "ENB"
]

# Fair PE pr. sektor (simple, konservativ)
FAIR_PE_BY_SECTOR = {
    "Technology": 22,
    "Consumer Defensive": 20,
    "Healthcare": 20,
    "Financial Services": 12,
    "Energy": 12,
    "Utilities": 16,
    "Real Estate": 16,
    "Industrials": 18,
    "Communication Services": 18,
    "Basic Materials": 14,
}

# ----------------------------
# HELPERS
# ----------------------------

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def calc_fair_value(eps, fair_pe):
    if eps is None or fair_pe is None:
        return None
    return eps * fair_pe


def calc_upside(price, fair_value):
    if price is None or fair_value is None or price <= 0:
        return None
    return (fair_value / price - 1) * 100


def calc_score(div_yield, upside, payout):
    """
    Score 0–100
    - Yield: bedst mellem 1–6%
    - Upside: cap ved 40%
    - Payout: straf hvis for høj
    """
    score = 50

    # Yield
    if div_yield is not None:
        if 1 <= div_yield <= 6:
            score += 20
        elif div_yield > 6:
            score += 10

    # Upside
    if upside is not None:
        score += min(max(upside, 0), 40) * 0.5  # max +20

    # Payout sanity
    if payout is not None:
        if payout > 100:
            score -= 20
        elif payout > 80:
            score -= 10

    return max(0, min(100, round(score, 1)))


def calc_signal(score, upside):
    if score >= 80 and upside is not None and upside > 15:
        return "GOLD"
    if score >= 65:
        return "BUY"
    if score >= 50:
        return "HOLD"
    return "WATCH"


def calc_confidence(eps, payout, div_yield):
    missing = sum(x is None for x in [eps, payout, div_yield])
    if missing == 0:
        return "High"
    if missing == 1:
        return "Medium"
    return "Low"


# ----------------------------
# MAIN
# ----------------------------

rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = safe_float(info.get("currentPrice"))
        dividend_yield = safe_float(info.get("dividendYield"))
        payout_ratio = safe_float(info.get("payoutRatio"))
        eps = safe_float(info.get("trailingEps"))
        pe = safe_float(info.get("trailingPE"))

        # Normaliser %
        if dividend_yield is not None:
            dividend_yield *= 100
        if payout_ratio is not None:
            payout_ratio *= 100

        sector = info.get("sector")
        fair_pe = FAIR_PE_BY_SECTOR.get(sector, 15)

        fair_value = calc_fair_value(eps, fair_pe)
        upside = calc_upside(price, fair_value)
        score = calc_score(dividend_yield, upside, payout_ratio)
        signal = calc_signal(score, upside)
        confidence = calc_confidence(eps, payout_ratio, dividend_yield)

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": sector,
            "Industry": info.get("industry"),
            "Price": price,
            "Dividend Yield (%)": round(dividend_yield, 2) if dividend_yield is not None else None,
            "Payout Ratio (%)": round(payout_ratio, 2) if payout_ratio is not None else None,
            "PE": pe,
            "FairValue": round(fair_value, 2) if fair_value is not None else None,
            "Upside (%)": round(upside, 2) if upside is not None else None,
            "Score": score,
            "Signal": signal,
            "Confidence": confidence
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)

# Sikr stabil kolonnerækkefølge (VIGTIGT for DataTables)
COLUMN_ORDER = [
    "Ticker", "Name", "Country", "Sector", "Industry",
    "Price", "Dividend Yield (%)", "Payout Ratio (%)", "PE",
    "FairValue", "Upside (%)", "Score", "Signal", "Confidence"
]

df = df[COLUMN_ORDER]

df.to_csv("screener_results.csv", index=False)
print(f"OK – wrote {len(df)} rows to screener_results.csv")
