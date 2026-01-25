import yfinance as yf
import csv
import os
from datetime import datetime

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_FILE = os.path.join(BASE_DIR, "tickers.txt")

# WRITE DIRECTLY INTO docs/ FOR GITHUB PAGES
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "data", "screener_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screener_results.csv")

FIELDS = [
    "GeneratedUTC",
    "Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Price",
    "DividendYield","PayoutRatio","PE",
    "Confidence","Signal",
    # DEBUG (so we can prove what's happening)
    "LTM_Dividend","LTM_Dividend_Count"
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
            # no inline comments
            t = line.split("#")[0].strip()
            if t:
                tickers.append(t)

    return sorted(set(tickers))


def safe_float(x, digits=2):
    try:
        return round(float(x), digits)
    except Exception:
        return ""


def calc_ltm_dividend_via_history(ticker_obj):
    """
    Robust: use history() and sum 'Dividends' over ~1 year.
    This works more reliably in GitHub Actions than ticker.dividends.
    """
    try:
        hist = ticker_obj.history(period="400d", interval="1d", actions=True, auto_adjust=False)
        if hist is None or hist.empty:
            return ("", 0)

        # Some tickers have no 'Dividends' column depending on data returned
        if "Dividends" not in hist.columns:
            return ("", 0)

        divs = hist["Dividends"].dropna()
        # Keep only actual dividend payments
        divs = divs[divs > 0]

        if divs.empty:
            return ("", 0)

        ltm_sum = float(divs.sum())
        return (round(ltm_sum, 4), int(divs.shape[0]))
    except Exception:
        return ("", 0)


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
    generated = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    print("▶ Dividend Screener starting")
    print("▶ GeneratedUTC:", generated)
    print("▶ Base dir:", BASE_DIR)
    print("▶ Loading tickers from:", TICKER_FILE)
    print("▶ Writing CSV to:", OUTPUT_FILE)

    tickers = load_tickers(TICKER_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            name = info.get("longName") or info.get("shortName") or ""
            price = info.get("regularMarketPrice")

            # If we can't even price it, skip
            if price is None:
                continue

            ltm_dividend, ltm_count = calc_ltm_dividend_via_history(t)

            dividend_yield = (
                round((ltm_dividend / float(price)) * 100, 2)
                if ltm_dividend != "" and float(price) > 0
                else ""
            )

            eps = info.get("trailingEps")
            payout_ratio = (
                round((ltm_dividend / float(eps)) * 100, 2)
                if ltm_dividend != "" and eps and float(eps) > 0
                else ""
            )

            pe = safe_float(info.get("trailingPE"), 4)

            confidence = calc_confidence(dividend_yield, payout_ratio, pe)
            signal = calc_signal(dividend_yield, payout_ratio)

            row = {
                "GeneratedUTC": generated,
                "Ticker": ticker,
                "Name": name,
                "Country": info.get("country",""),
                "Currency": info.get("currency",""),
                "Exchange": info.get("exchange",""),
                "Sector": info.get("sector",""),
                "Industry": info.get("industry",""),
                "Price": safe_float(price, 2),
                "DividendYield": dividend_yield,
                "PayoutRatio": payout_ratio,
                "PE": pe,
                "Confidence": confidence,
                "Signal": signal,
                "LTM_Dividend": ltm_dividend,
                "LTM_Dividend_Count": ltm_count
            }

            # Ensure all fields exist
            for f in FIELDS:
                row.setdefault(f, "")

            rows.append(row)

        except Exception as e:
            print(f"⚠ Skipped {ticker}: {e}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✔ CSV written: {OUTPUT_FILE}")
    print(f"✔ Rows: {len(rows)}")

if __name__ == "__main__":
    main()
