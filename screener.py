import yfinance as yf
import pandas as pd
import time
from datetime import datetime

TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "data/screener_results.csv"

def resolve_country(ticker, info):
    if info.get("country"):
        return info.get("country")

    if ticker.endswith(".CO"):
        return "Denmark"
    if ticker.endswith(".TO"):
        return "Canada"
    if ticker.endswith(".ST"):
        return "Sweden"
    if ticker.endswith(".HE"):
        return "Finland"
    if ticker.endswith(".AS"):
        return "Netherlands"
    if ticker.endswith(".L"):
        return "United Kingdom"

    if info.get("exchange") in ["NYQ", "NMS"]:
        return "United States"

    return None

def safe_float(value):
    try:
        return round(float(value), 2)
    except:
        return None

def fetch_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info or {}
        full = stock.info or {}

        price = safe_float(info.get("last_price"))
        dividend_yield = full.get("dividendYield")
        payout = full.get("payoutRatio")

        return {
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": ticker,
            "Name": full.get("shortName"),
            "Country": resolve_country(ticker, full),
            "Sector": full.get("sector"),
            "Industry": full.get("industry"),
            "Price": price,
            "DividendYield_%": safe_float(dividend_yield * 100) if dividend_yield else None,
            "PayoutRatio_%": safe_float(payout * 100) if payout else None,
            "PE": safe_float(full.get("trailingPE")),
        }

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        return None

def main():
    with open(TICKER_FILE) as f:
        tickers = [t.strip() for t in f if t.strip()]

    rows = []
    for t in tickers:
        print(f"Fetching {t}")
        row = fetch_ticker(t)
        if row:
            rows.append(row)
        time.sleep(2)  # 🔒 Rate-limit protection

    df = pd.DataFrame(rows)

    df = df[
        [
            "GeneratedUTC",
            "Ticker",
            "Name",
            "Country",
            "Sector",
            "Industry",
            "Price",
            "DividendYield_%",
            "PayoutRatio_%",
            "PE",
        ]
    ]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
