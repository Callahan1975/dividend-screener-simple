import pandas as pd
import yfinance as yf
from datetime import datetime
import time

TICKER_FILE = "data/tickers.txt"
OUTPUT_FILE = "docs/screener_results.csv"

def load_tickers():
    with open(TICKER_FILE) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]

def safe(val):
    return None if pd.isna(val) else val

def confidence_score(row):
    score = 50

    # Dividend yield
    if row["DividendYield_%"]:
        if row["DividendYield_%"] >= 3:
            score += 10
        elif row["DividendYield_%"] >= 2:
            score += 5

    # Payout ratio
    if row["PayoutRatio_%"]:
        if row["PayoutRatio_%"] < 60:
            score += 10
        elif row["PayoutRatio_%"] > 90:
            score -= 15

    # PE sanity
    if row["PE"]:
        if row["PE"] < 20:
            score += 5
        elif row["PE"] > 40:
            score -= 10

    return max(0, min(100, score))

def signal_from_confidence(c):
    if c >= 75:
        return "BUY"
    if c >= 55:
        return "HOLD"
    return "WATCH"

rows = []
for ticker in load_tickers():
    try:
        t = yf.Ticker(ticker)
        i = t.info

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Name": i.get("shortName"),
            "Country": i.get("country"),
            "Currency": i.get("currency"),
            "Exchange": i.get("exchange"),
            "Sector": i.get("sector"),
            "Industry": i.get("industry"),
            "Price": safe(i.get("currentPrice")),
            "DividendYield_%": safe((i.get("dividendYield") or 0) * 100),
            "PayoutRatio_%": safe((i.get("payoutRatio") or 0) * 100),
            "PE": safe(i.get("trailingPE")),
        })

        time.sleep(1)

    except Exception as e:
        print("ERROR", ticker, e)

df = pd.DataFrame(rows)

df["Confidence"] = df.apply(confidence_score, axis=1)
df["Signal"] = df["Confidence"].apply(signal_from_confidence)

df.to_csv(OUTPUT_FILE, index=False)
print("✅ screener_results.csv updated")
