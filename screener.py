import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import time

# =========================
# CONFIG (LÅST)
# =========================
TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "data/screener_results.csv"
REQUEST_SLEEP = 1.2  # rate-limit safe

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
def safe_float(v):
    try:
        if v is None or v == "":
            return ""
        return round(float(v), 2)
    except Exception:
        return ""

def get_country(info, ticker):
    # Prefer Yahoo country; fallback via suffix/exchange if missing
    c = info.get("country")
    if c:
        return c
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
    ex = info.get("exchange")
    if ex in ("NMS", "NYQ"):
        return "United States"
    return ""

def get_sector(info):
    return info.get("sector", "") or ""

# =========================
# CONFIDENCE MODEL (LÅST)
# =========================
def score_dividend(yield_pct):
    if yield_pct == "":
        return 0
    if yield_pct >= 4.0:
        return 30
    if yield_pct >= 2.5:
        return 22
    if yield_pct >= 1.0:
        return 14
    if yield_pct > 0:
        return 6
    return 0

def score_payout(payout):
    if payout == "":
        return 10  # neutral if missing
    if payout <= 50:
        return 30
    if payout <= 70:
        return 22
    if payout <= 90:
        return 10
    return 0

def score_pe(pe):
    if pe == "":
        return 10
    if pe <= 15:
        return 25
    if pe <= 20:
        return 20
    if pe <= 25:
        return 14
    if pe <= 30:
        return 8
    return 0

def score_sector(sector):
    s = sector.lower()
    if "utility" in s:
        return 15
    if "consumer defensive" in s:
        return 12
    if "health" in s:
        return 10
    if "financial" in s:
        return 8
    if "industrial" in s:
        return 6
    if "technology" in s:
        return 4
    return 2

def signal_from_confidence(score):
    if score >= 80:
        return "STRONG BUY"
    if score >= 65:
        return "BUY"
    if score >= 45:
        return "HOLD"
    if score >= 25:
        return "WATCH"
    return "AVOID"

# =========================
# MAIN LOOP
# =========================
rows = []
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

for ticker in tickers:
    print(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        price = info.get("currentPrice")
        annual_dividend = info.get("dividendRate")  # seneste årlige dividend

        # Dividend Yield = annual dividend / price
        dividend_yield = ""
        if price and annual_dividend and price > 0:
            dividend_yield = round((annual_dividend / price) * 100, 2)

        payout = info.get("payoutRatio")
        payout = round(payout * 100, 2) if payout is not None else ""

        pe = info.get("trailingPE")
        pe = round(pe, 2) if pe is not None else ""

        sector = get_sector(info)
        country = get_country(info, ticker)

        # Confidence
        confidence = (
            score_dividend(dividend_yield)
            + score_payout(payout)
            + score_pe(pe)
            + score_sector(sector)
        )

        row = {
            "GeneratedUTC": now_utc,
            "Ticker": ticker,
            "Name": info.get("shortName", "") or info.get("longName", ""),
            "Country": country,
            "Sector": sector,
            "Price": safe_float(price),
            "DividendYield": dividend_yield,
            "PayoutRatio": payout,
            "PE": pe,
            "Confidence": confidence,
            "Signal": signal_from_confidence(confidence),
        }

        rows.append(row)
        time.sleep(REQUEST_SLEEP)

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        rows.append({
            "GeneratedUTC": now_utc,
            "Ticker": ticker,
            "Name": "",
            "Country": "",
            "Sector": "",
            "Price": "",
            "DividendYield": "",
            "PayoutRatio": "",
            "PE": "",
            "Confidence": "",
            "Signal": "AVOID",
        })

# =========================
# SAVE CSV (KONTRAKT LÅST)
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
    "Confidence",
    "Signal",
])

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
