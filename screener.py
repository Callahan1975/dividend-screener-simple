import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime
from functools import lru_cache

# =========================
# KONFIG
# =========================
INPUT_TICKERS_FILE = "data/dividend_classes.csv"
OUTPUT_FILE = "docs/data/screener_results.csv"

REQUEST_SLEEP = 0.6  # beskytter mod rate limit

# =========================
# CACHE
# =========================
@lru_cache(maxsize=512)
def get_info_cached(ticker):
    return yf.Ticker(ticker).info

# =========================
# LOAD TICKERS
# =========================
tickers_df = pd.read_csv(INPUT_TICKERS_FILE)
tickers = tickers_df["Ticker"].dropna().unique().tolist()

# =========================
# LOAD GAMMEL DATA (fallback)
# =========================
old_df = None
if os.path.exists(OUTPUT_FILE):
    old_df = pd.read_csv(OUTPUT_FILE)

rows = []

# =========================
# LOOP
# =========================
for ticker in tickers:
    print(f"Processing {ticker}")
    time.sleep(REQUEST_SLEEP)

    try:
        info = get_info_cached(ticker)

        price = info.get("regularMarketPrice")
        dividend_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")
        pe = info.get("trailingPE")

        country = info.get("country")
        sector = info.get("sector")
        currency = info.get("currency")
        exchange = info.get("exchange")
        industry = info.get("industry")
        name = info.get("shortName") or info.get("longName")

        # Konverter til procent
        if dividend_yield is not None:
            dividend_yield = round(dividend_yield * 100, 2)

        if payout_ratio is not None:
            payout_ratio = round(payout_ratio * 100, 2)

    except Exception as e:
        print(f"Rate limit / error on {ticker}: {e}")

        if old_df is not None and ticker in old_df["Ticker"].values:
            prev = old_df[old_df["Ticker"] == ticker].iloc[0]
            price = prev["Price"]
            dividend_yield = prev["DividendYield_%"]
            payout_ratio = prev["PayoutRatio_%"]
            pe = prev["PE"]
            country = prev["Country"]
            sector = prev["Sector"]
            currency = prev["Currency"]
            exchange = prev["Exchange"]
            industry = prev["Industry"]
            name = prev["Name"]
        else:
            price = None
            dividend_yield = None
            payout_ratio = None
            pe = None
            country = None
            sector = None
            currency = None
            exchange = None
            industry = None
            name = None

    rows.append({
        "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Name": name,
        "Country": country,
        "Currency": currency,
        "Exchange": exchange,
        "Sector": sector,
        "Industry": industry,
        "Price": price,
        "DividendYield_%": dividend_yield,
        "PayoutRatio_%": payout_ratio,
        "PE": pe
    })

# =========================
# WRITE OUTPUT
# =========================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df = pd.DataFrame(rows)

# Sikrer kolonneorden
df = df[
    [
        "GeneratedUTC",
        "Ticker",
        "Name",
        "Country",
        "Currency",
        "Exchange",
        "Sector",
        "Industry",
        "Price",
        "DividendYield_%",
        "PayoutRatio_%",
        "PE",
    ]
]

df.to_csv(OUTPUT_FILE, index=False)
print("✅ Screener completed successfully")
