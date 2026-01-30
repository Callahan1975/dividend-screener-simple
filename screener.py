import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# -------------------------
# CONFIG
# -------------------------
TICKERS = [
    "AAPL", "JNJ", "MSFT", "KO", "PG"
]

OUTPUT_PATH = "data/screener_results.csv"

MAX_REASONABLE_YIELD = 15.0   # %
MAX_REASONABLE_PAYOUT = 110.0 # %

# -------------------------
# HELPERS
# -------------------------
def normalize_dividend_yield(raw):
    """
    Normalize dividend yield to %.
    Accepts either ratio (0.041) or percent (4.1).
    Rejects absurd values.
    """
    try:
        if raw is None:
            return None
        y = float(raw)
    except Exception:
        return None

    # ratio → %
    if y <= 1:
        y = y * 100

    # sanity
    if y <= 0 or y > MAX_REASONABLE_YIELD:
        return None

    return round(y, 2)


def normalize_payout_ratio(raw):
    try:
        if raw is None:
            return None
        p = float(raw)
    except Exception:
        return None

    if p <= 0 or p > MAX_REASONABLE_PAYOUT:
        return None

    return round(p, 2)


def safe_float(v):
    try:
        return round(float(v), 4)
    except Exception:
        return None


# -------------------------
# MAIN
# -------------------------
rows = []

generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

for ticker in TICKERS:
    print(f"Processing {ticker}...")
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        continue

    flags = []

    price = safe_float(info.get("currentPrice"))
    dividend_yield = normalize_dividend_yield(info.get("dividendYield"))
    payout_ratio = normalize_payout_ratio(info.get("payoutRatio"))

    if dividend_yield is None:
        flags.append("YieldMissingOrInvalid")

    if payout_ratio is None:
        flags.append("PayoutMissingOrInvalid")

    row = {
        "GeneratedUTC": generated_utc,
        "Ticker": ticker,
        "Name": info.get("shortName"),
        "Country": info.get("country"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Price": price,
        "DividendYield_%": dividend_yield,
        "PayoutRatio_%": payout_ratio,
        "PE": safe_float(info.get("trailingPE")),
        "Flags": ";".join(flags) if flags else ""
    }

    rows.append(row)

# -------------------------
# EXPORT
# -------------------------
df = pd.DataFrame(rows)

# Column order (explicit & stable)
ordered_columns = [
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
    "Flags"
]

df = df.reindex(columns=ordered_columns)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
