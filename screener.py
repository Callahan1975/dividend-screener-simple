import yfinance as yf
import csv
import os
from datetime import datetime

# =============================
# PATHS (VIGTIGT)
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_FILE = os.path.join(BASE_DIR, "tickers.txt")

# ⬇️ WRITE DIRECTLY TO DOCS FOR GITHUB PAGES
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "data", "screener_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screener_results.csv")

FIELDS = [
    "Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Price",
    "DividendYield","PayoutRatio","PE",
    "Confidence","Signal"
]

# =============================
# HELPERS
# =============================

def load_tickers(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ticker file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return sorted({
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        })


def safe_float(x, digits=2):
    try:
        return round(float(x), digits)
    except Exception:
        return ""


def calc_ltm_dividend(t):
    divs = t.dividends
    if divs is None or divs.empty:
        return ""
    ltm = divs.last("365D")
    if ltm.empty:
        return ""
    return round(float(ltm.sum()), 4)


def calc_signal(yield_pct, payout):
    if yield_pct == "" or payout == "":
        return "HOLD"
    if yield_pct >= 4 and payout <= 70:
        return "BUY"
    if payout > 90:
        return "AVOID"
    return "HOLD"


def calc_confidence(yield_pct, payout, pe):
    score = 50
    if yield_pct != "" and yield_pct >= 3:
        score += 15
    if payout != "" and payout <= 70:
        score += 15
    if pe != "" and pe <= 20:
        score += 10
    return min(score, 100)

# =============================
# MAIN
# =============================

def main():
    print("▶ Dividend Screener starting")

    tickers = load_tickers(TICKER_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            name = info.get("longName") or info.get("shortName")
            price = info.get("regularMarketPrice")

            if not name or price is None:
                continue

            ltm_dividend = calc_ltm_dividend(t)

            dividend_yield = (
                round((ltm_dividend / price) * 100, 2)
                if ltm_dividend != ""
                else ""
            )

            eps = info.get("trailingEps")
            payout_ratio = (
                round((ltm_dividend / eps) * 100, 2)
                if ltm_dividend != "" and eps and eps > 0
                else ""
            )

            pe = safe_float(info.get("trailingPE"))

            confidence = calc_confidence(dividend_yield, payout_ratio, pe)
            signal = calc_signal(dividend_yield, payout_ratio)

            rows.append({
                "Ticker": ticker,
                "Name": name,
                "Country": info.get("country",""),
                "Currency": info.get("currency",""),
                "Exchange": info.get("exchange",""),
                "Sector": info.get("sector",""),
                "Industry": info.get("industry",""),
                "Price": safe_float(price),
                "DividendYield": dividend_yield,
                "PayoutRatio": payout_ratio,
                "PE": pe,
                "Confidence": confidence,
                "Signal": signal
            })

        except Exception as e:
            print(f"⚠ Skipped {ticker}: {e}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✔ CSV written to: {OUTPUT_FILE}")
    print(f"✔ Rows: {len(rows)}")
    print(f"✔ {datetime.utcnow().isoformat()}Z")

if __name__ == "__main__":
    main()
