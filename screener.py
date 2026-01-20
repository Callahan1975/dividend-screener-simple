import yfinance as yf
import pandas as pd
from datetime import datetime

# --------------------
# Helpers
# --------------------
def safe(val, default=0):
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

# --------------------
# Load tickers
# --------------------
with open("ticker_live.txt", "r") as f:
    tickers = [t.strip() for t in f if t.strip()]

rows = []

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        dividend_yield = safe(info.get("dividendYield")) * 100
        payout_ratio = safe(info.get("payoutRatio")) * 100
        pe = safe(info.get("trailingPE"))

        country = info.get("country", "")
        sector = info.get("sector", "")
        currency = info.get("currency", "")

        # --------------------
        # Simple scoring (robust)
        # --------------------
        score = 0

        if dividend_yield >= 3:
            score += 30
        elif dividend_yield >= 1.5:
            score += 15

        if payout_ratio > 0 and payout_ratio <= 75:
            score += 25
        elif payout_ratio <= 100:
            score += 10

        if pe > 0 and pe <= 18:
            score += 25
        elif pe <= 25:
            score += 10

        confidence = min(score, 100)

        if confidence >= 75:
            signal = "BUY"
        elif confidence >= 55:
            signal = "HOLD"
        else:
            signal = "WATCH"

        why = f"Yield {dividend_yield:.1f}%, payout {payout_ratio:.0f}%, PE {pe:.1f}"

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": country,
            "Sector": sector,
            "Currency": currency,
            "Price": round(price, 2),
            "DividendYield": round(dividend_yield, 2),
            "PayoutRatio": round(payout_ratio, 2),
            "PE": round(pe, 2),
            "Score": score,
            "Signal": signal,
            "Confidence": confidence,
            "Why": why
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

# --------------------
# Save CSV
# --------------------
df = pd.DataFrame(rows)

output_path = "data/screener_results/screener_results.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} rows to {output_path}")
