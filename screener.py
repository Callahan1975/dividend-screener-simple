import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import math

TICKERS = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "TROW", "ENB"
]

SECTOR_FAIR_PE = {
    "Technology": 22,
    "Consumer Defensive": 20,
    "Healthcare": 18,
    "Financial Services": 12,
    "Energy": 12,
    "Utilities": 16,
    "Real Estate": 16,
}

def clamp(val, lo, hi):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return max(lo, min(val, hi))

rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice")
        eps = info.get("trailingEps")
        pe = info.get("trailingPE")
        sector = info.get("sector")

        fair_pe_default = SECTOR_FAIR_PE.get(sector, 15)
        fair_pe = clamp(pe, 8, fair_pe_default) if pe else fair_pe_default

        fair_value = eps * fair_pe if eps and fair_pe else None
        upside = ((fair_value / price - 1) * 100) if fair_value and price else None

        dividend_yield = (info.get("dividendYield") or 0) * 100
        payout = (info.get("payoutRatio") or 0) * 100

        # --- SCORING ---
        score = 0
        flags = []

        # Yield (0–25)
        if 1 <= dividend_yield <= 6:
            score += 25
        elif dividend_yield > 0:
            score += 10
        else:
            flags.append("NO_DIVIDEND")

        # Payout (0–25)
        payout_limit = 110 if sector in ["Real Estate", "Financial Services"] else 90
        if payout <= payout_limit:
            score += 25
        else:
            score += 5
            flags.append("HIGH_PAYOUT")

        # Valuation (0–25)
        if upside is not None:
            if upside > 20:
                score += 25
            elif upside > 0:
                score += 15
            else:
                score += 5
        else:
            flags.append("NO_FAIR_VALUE")

        # Quality (0–25) – simpel baseline
        if pe and pe < fair_pe_default:
            score += 25
        else:
            score += 10

        # Signal
        if score >= 85 and (upside or 0) > 10:
            signal = "GOLD"
        elif score >= 70:
            signal = "BUY"
        elif score >= 50:
            signal = "HOLD"
        else:
            signal = "WATCH"

        confidence = "High" if len(flags) == 0 else ("Medium" if len(flags) == 1 else "Low")

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": sector,
            "Industry": info.get("industry"),
            "Price": price,
            "Dividend Yield (%)": round(dividend_yield, 2),
            "Payout Ratio (%)": round(payout, 2),
            "PE": round(pe, 2) if pe else None,
            "FairPE": round(fair_pe, 2) if fair_pe else None,
            "FairValue": round(fair_value, 2) if fair_value else None,
            "Upside (%)": round(upside, 2) if upside else None,
            "Score": score,
            "Signal": signal,
            "Confidence": confidence,
            "Flags": ",".join(flags)
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)
df.to_csv("screener_results.csv", index=False)
print(f"Saved {len(df)} rows to screener_results.csv")
