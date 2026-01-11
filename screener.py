import pandas as pd
import yfinance as yf

df = pd.read_csv("input.csv")

results = []
for ticker in df["ticker"]:
    stock = yf.Ticker(ticker)
    info = stock.info
    results.append({
        "ticker": ticker,
        "dividendYield": info.get("dividendYield"),
        "payoutRatio": info.get("payoutRatio"),
        "marketCap": info.get("marketCap"),
    })

out = pd.DataFrame(results)
out.to_csv("output.csv", index=False)
print("Saved output.csv")
