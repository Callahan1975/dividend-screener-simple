import pandas as pd
import yfinance as yf
from datetime import datetime
import os

TICKERS = [
    "AAPL",
    "MSFT",
    "JNJ",
    "PG",
    "KO",
    "PEP",
    "TROW",
    "ENB"
]

rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": info.get("currentPrice"),
            "Dividend Yield (%)": (info.get("dividendYield") or 0) * 100,
            "Payout Ratio (%)": (info.get("payoutRatio") or 0) * 100,
            "PE": info.get("trailingPE")
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)
df.to_csv("screener_results.csv", index=False)
