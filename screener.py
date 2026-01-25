import yfinance as yf
import csv
import os
from datetime import datetime

# =============================
# PATHS (ABSOLUT – STABILT)
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TICKER_FILE = os.path.join(BASE_DIR, "tickers.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "screener_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screener_results.csv")

FIELDS = [
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

# =============================
# HELPERS
# =============================

def load_tickers(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ticker file not found: {path}")

    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ticker = line.split("#")[0].strip()
            if ticker:
                tickers.append(ticker)

    return sorted(set(tickers))


def safe_float(x, digits=4):
    try:
        if x is None:
            return ""
        return round(float(x), digits)
    except Exception:
        return ""


def calc_signal(yield_pct, payout):
    if yield_pct == "" or payout == "":
        return "HOLD"
    if yield_pct >= 4 and payout <= 70:
        return "BUY"
    if payout > 90:
        return "SELL"
    return "HOLD"


def calc_confidence(yield_pct, payout, pe):
    score = 50
    if yield_pct != "" and yield_pct >= 3:
        score += 10
    if payout != "" and payout <= 70:
        score += 10
    if pe != "" and pe <= 20:
        score += 10
    return min(score, 100)

# =============================
# MAIN
# =============================

def main():
    print("▶ Dividend Screener starting")
    print(f"▶ Base dir: {BASE_DIR}")
    print(f"▶ Loading tickers from: {TICKER_FILE}")

    tickers = load_tickers(TICKER_FILE)

    if not tickers:
        raise RuntimeError("No valid tickers found")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            name = info.get("longName") or info.get("shortName")
            price = info.get("regularMarketPrice")

            # Drop broken rows
            if not name or price is None:
                continue

            dividend_yield = safe_float(
                info.get("dividendYield") * 100 if info.get("dividendYield") else ""
            )
            payout_ratio = safe_float(
                info.get("payoutRatio") * 100 if info.get("payoutRatio") else ""
            )
            pe = safe_float(info.get("trailingPE"))

            confidence = calc_confidence(dividend_yield, payout_ratio, pe)
            signal = calc_signal(dividend_yield, payout_ratio)

            row = {
                "Ticker": ticker,
                "Name": name,
                "Country": info.get("country", ""),
                "Currency": info.get("currency", ""),
                "Exchange": info.get("exchange", ""),
                "Sector": info.get("sector", ""),
                "Industry": info.get("industry", ""),
                "Price": safe_float(price),
                "DividendYield": dividend_yield,
                "PayoutRatio": payout_ratio,
                "PE": pe,
                "Confidence": confidence,
                "Signal": signal
            }

            for f in FIELDS:
                row.setdefault(f, "")

            rows.append(row)

        except Exception as e:
            print(f"⚠ Skipped {ticker}: {e}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✔ CSV written: {OUTPUT_FILE}")
    print(f"✔ Rows: {len(rows)}")
    print(f"✔ Timestamp: {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
