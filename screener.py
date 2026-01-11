import pandas as pd
import yfinance as yf

tickers = pd.read_csv("input.csv")["ticker"].dropna().astype(str).str.strip().unique()

rows = []

for ticker in tickers:
    t = yf.Ticker(ticker)
    info = t.info or {}

    market_cap = info.get("marketCap")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    div_y = info.get("dividendYield")          # decimal (0.025 = 2.5%)
    payout = info.get("payoutRatio")           # decimal (0.50 = 50%)

    rows.append({
        "Ticker": ticker,
        "Name": info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Currency": info.get("currency"),

        "Price": price,
        "Market Cap (USD bn)": None if market_cap is None else round(market_cap / 1e9, 1),

        "Dividend Yield (%)": None if div_y is None else round(div_y * 100, 2),
        "Dividend Rate (annual)": info.get("dividendRate"),
        "Payout Ratio (%)": None if payout is None else round(payout * 100, 2),

        "PE (TTM)": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "PEG": info.get("pegRatio"),

        "EPS (TTM)": info.get("trailingEps"),
        "EPS Forward": info.get("forwardEps"),

        "Revenue Growth": info.get("revenueGrowth"),
        "Earnings Growth": info.get("earningsGrowth"),

        "Profit Margin": info.get("profitMargins"),
        "Operating Margin": info.get("operatingMargins"),

        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),

        "Debt/Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),

        "Free Cash Flow": info.get("freeCashflow"),
        "Operating Cash Flow": info.get("operatingCashflow"),

        "Beta": info.get("beta"),

        "52w Low": info.get("fiftyTwoWeekLow"),
        "52w High": info.get("fiftyTwoWeekHigh"),
    })

out = pd.DataFrame(rows)
out.to_csv("output.csv", index=False)

print(f"Saved output.csv with {len(out)} rows and {len(out.columns)} columns")
