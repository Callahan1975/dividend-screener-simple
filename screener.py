import pandas as pd
import yfinance as yf

df = pd.read_csv("input.csv")

rows = []

for ticker in df["ticker"]:
    stock = yf.Ticker(ticker)
    info = stock.info

    dividend_yield = info.get("dividendYield")
    payout = info.get("payoutRatio")
    market_cap = info.get("marketCap")

    if dividend_yield is None or payout is None or market_cap is None:
        continue

    rows.append({
        "Ticker": ticker,
        "Dividend Yield (%)": round(dividend_yield * 100, 2),
        "Payout Ratio (%)": round(payout * 100, 2),
        "Market Cap (USD bn)": round(market_cap / 1e9, 1),
    })

out = pd.DataFrame(rows)

# SIMPLE FILTRE (kan justeres)
out = out[
    (out["Dividend Yield (%)"] >= 2) &
    (out["Payout Ratio (%)"] <= 60) &
    (out["Market Cap (USD bn)"] >= 50)
]

out.to_csv("output.csv", index=False)
print("Saved filtered output.csv")

