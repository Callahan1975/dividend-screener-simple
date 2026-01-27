import pandas as pd
import yfinance as yf
from datetime import datetime
import os

TICKERS = [
    "AAPL",
    "JNJ",
    "MSFT",
    "KO",
    "PG"
]

rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": info.get("currentPrice"),
            "DividendYield_%": (info.get("dividendYield") or 0) * 100,
            "PayoutRatio_%": (info.get("payoutRatio") or 0) * 100,
            "PE": info.get("trailingPE"),
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/screener_results.csv", index=False)
