import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# =========================
# CONFIG
# =========================
TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "data/screener_results.csv"
SLEEP_BETWEEN_CALLS = 1.2  # rate limit protection

COLUMNS = [
    "GeneratedUTC",
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Price",
    "DividendYield_%",
    "PayoutRatio_%",
    "PE"
]

# =========================
# LOAD TICKERS
# =========================
with open(TICKER_FILE, "r") as f:
    tickers = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

rows = []
now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# =========================
# FETCH DATA
# =========================
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info if hasattr(stock, "fast_info") else {}
        full = stock.info or {}

        price = info.get("last_price") or full.get("currentPrice")

        dividend_yield = full.get("dividendYield")
        if dividend_yield is not None:
            dividend_yield = round(dividend_yield * 100, 2)

        payout = full.get("payoutRatio")
        if payout is not None:
            payout = round(payout * 100, 2)

        pe = full.get("trailingPE")
        if pe is not None:
            pe = round(pe, 2)

        row = {
            "GeneratedUTC": now_utc,
            "Ticker": ticker,
            "Name": full.get("shortName") or full.get("longName"),
            "Country": full.get("country"),
            "Sector": full.get("sector"),
            "Price": round(price, 2) if price else None,
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout,
            "PE": pe,
        }

        rows.append(row)
        time.sleep(SLEEP_BETWEEN_CALLS)

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        rows.append({
            "GeneratedUTC": now_utc,
            "Ticker": ticker,
            "Name": None,
            "Country": None,
            "Sector": None,
            "Price": None,
            "DividendYield_%": None,
            "PayoutRatio_%": None,
            "PE": None,
        })

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Screener completed: {len(df)} tickers written to {OUTPUT_FILE}")
