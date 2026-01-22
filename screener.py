import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

TICKER_FILE = "data/tickers.txt"
OUTPUT_CSV = "data/screener_results.csv"

GOLD_HEADERS = [
    "Ticker",
    "Name",
    "Country",
    "Currency",
    "Exchange",
    "Sector",
    "Industry",
    "Price",
    "DividendYield",
    "PayoutRatio",
    "PE",
    "Confidence",
    "Signal"
]

# =========================================================
# HELPERS
# =========================================================

def load_tickers(path):
    with open(path, "r") as f:
        return [t.strip() for t in f if t.strip() and not t.startswith("#")]

def safe(val):
    return None if val in [None, "", "nan"] else val

def calc_confidence(row):
    score = 50

    if row["DividendYield"] and row["DividendYield"] >= 0.03:
        score += 10
    if row["PayoutRatio"] and row["PayoutRatio"] <= 0.7:
        score += 10
    if row["PE"] and row["PE"] <= 20:
        score += 10

    return min(score, 100)

def calc_signal(confidence):
    if confidence >= 75:
        return "BUY"
    if confidence >= 60:
        return "HOLD"
    return "WATCH"

def validate_headers(df):
    actual = list(df.columns)
    if actual != GOLD_HEADERS:
        raise ValueError(
            f"""
CSV HEADER MISMATCH 🚨

Expected:
{GOLD_HEADERS}

Actual:
{actual}
"""
        )

# =========================================================
# MAIN
# =========================================================

rows = []
tickers = load_tickers(TICKER_FILE)

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("dividendYield"))
        payout = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        row = {
            "Ticker": ticker,
            "Name": safe(info.get("shortName")),
            "Country": safe(info.get("country")),
            "Currency": safe(info.get("currency")),
            "Exchange": safe(info.get("exchange")),
            "Sector": safe(info.get("sector")),
            "Industry": safe(info.get("industry")),
            "Price": price,
            "DividendYield": dividend,
            "PayoutRatio": payout,
            "PE": pe,
            "Confidence": None,
            "Signal": None,
        }

        row["Confidence"] = calc_confidence(row)
        row["Signal"] = calc_signal(row["Confidence"])

        rows.append(row)

    except Exception as e:
        print(f"⚠️ Skipped {ticker}: {e}")

# =========================================================
# DATAFRAME + VALIDATION
# =========================================================

df = pd.DataFrame(rows)

# FORCE COLUMN ORDER (CRITICAL)
df = df[GOLD_HEADERS]

# VALIDATE BEFORE WRITE
validate_headers(df)

# WRITE CSV
df.to_csv(OUTPUT_CSV, index=False)

print("✅ CSV generated successfully")
print(f"Rows: {len(df)}")
print("✅ Headers locked & validated")
