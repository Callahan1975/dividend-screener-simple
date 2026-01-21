import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# =========================
# CONFIG
# =========================
TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "data/screener_results.csv"
SLEEP = 1.2

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
with open(TICKER_FILE) as f:
    tickers = [l.strip() for l in f if l.strip() and not l.startswith("#")]

rows = []
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# =========================
# FETCH DATA
# =========================
for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        price = info.get("currentPrice")
        dividend_rate = info.get("trailingAnnualDividendRate")
        payout = info.get("payoutRatio")
        pe = info.get("trailingPE")

        # ✅ CORRECT dividend yield calculation
        dividend_yield = None
        if price and dividend_rate:
            dividend_yield = round((dividend_rate / price) * 100, 2)

        row = {
            "GeneratedUTC": now,
            "Ticker": ticker,
            "Name": info.get("longName") or info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Price": round(price, 2) if price else None,
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": round(payout * 100, 2) if payout else None,
            "PE": round(pe, 2) if pe else None,
        }

        rows.append(row)
        time.sleep(SLEEP)

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        rows.append({
            "GeneratedUTC": now,
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
# SAVE
# =========================
df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Screener finished: {len(df)} tickers")
