import yfinance as yf
import pandas as pd
from datetime import datetime

TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "docs/screener_results.csv"

COLUMNS = [
    "GeneratedUTC",
    "Ticker",
    "Name",
    "Country",
    "Currency",
    "Exchange",
    "Sector",
    "Industry",
    "Price",
    "DividendYield",
    "PayoutRatio",
    "PE",
    "Confidence",
    "Signal"
]

rows = []

with open(TICKER_FILE) as f:
    tickers = [l.strip() for l in f if l.strip() and not l.startswith("#")]

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = info.get("currentPrice")
        dividend = info.get("dividendYield")
        payout = info.get("payoutRatio")

        dividend_yield = round(dividend * 100, 2) if dividend else None
        payout_ratio = round(payout * 100, 2) if payout else None

        confidence = 50
        if dividend_yield and dividend_yield > 2:
            confidence += 10
        if payout_ratio and payout_ratio < 70:
            confidence += 10

        if confidence >= 75:
            signal = "BUY"
        elif confidence >= 60:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Name": info.get("longName"),
            "Country": info.get("country"),
            "Currency": info.get("currency"),
            "Exchange": info.get("exchange"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": round(price, 2) if price else None,
            "DividendYield": dividend_yield,
            "PayoutRatio": payout_ratio,
            "PE": info.get("trailingPE"),
            "Confidence": confidence,
            "Signal": signal
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
