import yfinance as yf
import pandas as pd

tickers = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP",
    "V", "MA", "AVGO", "UNH", "NEE"
]

rows = []

for t in tickers:
    try:
        stock = yf.Ticker(t)
        info = stock.info

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName"),
            "Sector": info.get("sector"),
            "Price": info.get("currentPrice"),
            "DividendYield_%": round((info.get("dividendYield") or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio") or 0) * 100, 2),
            "PE": info.get("trailingPE"),
            "Signal": "BUY" if (info.get("dividendYield") or 0) > 0.02 else "HOLD"
        })
    except Exception as e:
        print(f"Error on {t}: {e}")

df = pd.DataFrame(rows)
df.to_csv("docs/screener_results.csv", index=False)
print("✅ screener_results.csv generated")
