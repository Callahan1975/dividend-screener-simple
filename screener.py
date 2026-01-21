import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

# ==============================
# CONFIG
# ==============================
TICKER_FILE = Path("data/tickers.txt")
OUTPUT_FILE = Path("data/screener_results.csv")

SLEEP_SECONDS = 2          # pause mellem tickers
MAX_RETRIES = 3            # retry ved rate limit
RETRY_SLEEP = 10           # pause ved rate limit

# ==============================
# HELPERS
# ==============================
def safe_pct(value):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value) * 100, 2)
    except Exception:
        return None

def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 2)
    except Exception:
        return None

def fetch_ticker_data(ticker):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            info_full = t.info

            price = safe_float(info.get("last_price"))

            dividend_yield = safe_pct(
                info_full.get("dividendYield")
                or info_full.get("trailingAnnualDividendYield")
            )

            payout_ratio = safe_pct(info_full.get("payoutRatio"))
            pe = safe_float(info_full.get("trailingPE"))

            return {
                "Ticker": ticker,
                "Name": info_full.get("shortName"),
                "Country": info_full.get("country"),
                "Sector": info_full.get("sector"),
                "Industry": info_full.get("industry"),
                "Price": price,
                "DividendYield_%": dividend_yield,
                "PayoutRatio_%": payout_ratio,
                "PE": pe,
            }

        except Exception as e:
            if "Too Many Requests" in str(e):
                print(f"RATE LIMIT {ticker} – retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_SLEEP)
            else:
                print(f"ERROR {ticker}: {e}")
                break

    # fallback – ALDRIG 0.00
    return {
        "Ticker": ticker,
        "Name": None,
        "Country": None,
        "Sector": None,
        "Industry": None,
        "Price": None,
        "DividendYield_%": None,
        "PayoutRatio_%": None,
        "PE": None,
    }

# ==============================
# MAIN
# ==============================
def main():
    if not TICKER_FILE.exists():
        raise FileNotFoundError("tickers.txt not found")

    tickers = [
        t.strip()
        for t in TICKER_FILE.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]

    rows = []
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for ticker in tickers:
        print(f"Fetching {ticker}")
        data = fetch_ticker_data(ticker)
        data["GeneratedUTC"] = generated_utc
        rows.append(data)
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(rows)

    # FAST kolonneorden – må IKKE ændres
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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
