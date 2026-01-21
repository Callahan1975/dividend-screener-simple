import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# =========================
# CONFIG
# =========================
TICKER_FILE = "data/ticker_alias.csv"
OUTPUT_FILE = "data/screener_results.csv"

SLEEP_BETWEEN_CALLS = 1.2  # beskytter mod rate limit


# =========================
# LOAD TICKERS
# =========================
tickers_df = pd.read_csv(TICKER_FILE)
tickers = tickers_df["Ticker"].dropna().unique().tolist()


# =========================
# HELPERS
# =========================
def safe_float(x):
    try:
        return float(x)
    except:
        return None


def confidence_score(row):
    score = 0

    # ---- 1. Valuation (PE)
    pe = safe_float(row.get("PE"))
    if pe is not None:
        if pe <= 15:
            score += 2
        elif pe <= 25:
            score += 1
        elif pe <= 35:
            score += 0
        else:
            score -= 1

    # ---- 2. Dividend presence
    dy = safe_float(row.get("DividendYield_%"))
    if dy is not None and dy > 0:
        score += 1

    # ---- 3. Payout discipline
    payout = safe_float(row.get("PayoutRatio_%"))
    if payout is not None:
        if payout <= 60:
            score += 2
        elif payout <= 80:
            score += 1
        elif payout <= 100:
            score += 0
        else:
            score -= 2

    # ---- 4. Sector stability
    sector = str(row.get("Sector", "")).lower()
    if sector in ["utilities", "consumer defensive", "healthcare"]:
        score += 1

    return score


def signal_from_score(score):
    if score >= 5:
        return "STRONG"
    elif score >= 3:
        return "GOOD"
    elif score >= 1:
        return "OK"
    elif score == 0:
        return "NEUTRAL"
    else:
        return "RISK"


# =========================
# MAIN LOOP
# =========================
rows = []
generated_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

for ticker in tickers:
    print(f"Processing {ticker}")
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        row = {
            "GeneratedUTC": generated_utc,
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Currency": info.get("currency"),
            "Exchange": info.get("exchange"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": safe_float(info.get("currentPrice")),
            "DividendYield_%": safe_float(
                info.get("dividendYield") * 100
                if info.get("dividendYield") is not None
                else None
            ),
            "PayoutRatio_%": safe_float(
                info.get("payoutRatio") * 100
                if info.get("payoutRatio") is not None
                else None
            ),
            "PE": safe_float(info.get("trailingPE")),
        }

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        row = {
            "GeneratedUTC": generated_utc,
            "Ticker": ticker,
            "Name": None,
            "Country": None,
            "Currency": None,
            "Exchange": None,
            "Sector": None,
            "Industry": None,
            "Price": None,
            "DividendYield_%": None,
            "PayoutRatio_%": None,
            "PE": None,
        }

    rows.append(row)
    time.sleep(SLEEP_BETWEEN_CALLS)


# =========================
# BUILD DATAFRAME
# =========================
df = pd.DataFrame(rows)

# ---- Confidence + Signal
df["ConfidenceScore"] = df.apply(confidence_score, axis=1)
df["Signal"] = df["ConfidenceScore"].apply(signal_from_score)

# ---- Sort for readability
df = df.sort_values(
    by=["ConfidenceScore", "DividendYield_%"],
    ascending=[False, False],
    na_position="last",
)

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print("✅ screener_results.csv written")
