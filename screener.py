import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import time

# =========================
# CONFIG
# =========================
TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "data/screener_results.csv"

REQUEST_SLEEP = 1.2  # seconds between requests (rate limit safe)

# =========================
# LOAD TICKERS
# =========================
with open(TICKER_FILE, "r") as f:
    tickers = [
        line.strip()
        for line in f.readlines()
        if line.strip() and not line.startswith("#")
    ]

# =========================
# HELPERS
# =========================
def safe_float(value):
    try:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return round(value, 2)
        return float(value)
    except Exception:
        return ""

def get_country(info):
    country = info.get("country")
    if not country:
        return ""
    return country

def get_sector(info):
    return info.get("sector", "") or ""

# =========================
# MAIN LOOP
# =========================
rows = []

for ticker in tickers:
    print(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice")
        annual_dividend = info.get("dividendRate")

        # Dividend Yield (annual dividend / price)
        dividend_yield = ""
        if price and annual_dividend and price > 0:
            dividend_yield = round((annual_dividend / price) * 100, 2)

        payout_ratio = info.get("payoutRatio")
        if payout_ratio is not None:
            payout_ratio = round(payout_ratio * 100, 2)

        pe = info.get("trailingPE")

        row = {
            "GeneratedUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": get_country(info),
            "Sector": get_sector(info),
            "Price": safe_float(price),
            "DividendYield": dividend_yield,
            "PayoutRatio": payout_ratio if payout_ratio is not None else "",
            "PE": safe_float(pe),
        }

        rows.append(row)

        time.sleep(REQUEST_SLEEP)

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        rows.append({
            "GeneratedUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": ticker,
            "Name": "",
            "Country": "",
            "Sector": "",
            "Price": "",
            "DividendYield": "",
            "PayoutRatio": "",
            "PE": "",
        })

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(rows, columns=[
    "GeneratedUTC",
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Price",
    "DividendYield",
    "PayoutRatio",
    "PE",
])

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
