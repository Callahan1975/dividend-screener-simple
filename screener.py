import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG
# =========================

TICKER_FILE = "data/screener_results/tickers_master.txt"
OUTPUT_FILE = "data/screener_results/screener_results.csv"

# fallback hvis tickers_master.txt mangler
FALLBACK_TICKERS = [
    "NOVO-B.CO", "DSV.CO", "MAERSK-B.CO", "CARL-B.CO", "PNDORA.CO",
    "COLO-B.CO", "DANSKE.CO", "SYDB.CO", "TRYG.CO",
    "AAPL", "MSFT", "KO", "PEP", "PG"
]

# =========================
# LOAD TICKERS
# =========================

if Path(TICKER_FILE).exists():
    with open(TICKER_FILE, "r") as f:
        tickers = [t.strip() for t in f if t.strip() and not t.startswith("#")]
else:
    print("⚠️ tickers_master.txt not found – using fallback list")
    tickers = FALLBACK_TICKERS

# =========================
# HELPERS
# =========================

def safe(info, key):
    val = info.get(key)
    if val in [None, "", "N/A"]:
        return None
    return val

def pct(val):
    try:
        return round(float(val) * 100, 2)
    except:
        return None

# =========================
# MAIN LOOP
# =========================

rows = []

for ticker in tickers:
    print(f"Fetching {ticker}...")
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info, "currentPrice") or safe(info, "regularMarketPrice")
        dividend_yield = pct(safe(info, "dividendYield"))
        payout_ratio = pct(safe(info, "payoutRatio"))

        row = {
            "Ticker": ticker,
            "Name": safe(info, "shortName"),
            "Country": safe(info, "country"),
            "Sector": safe(info, "sector"),
            "Currency": safe(info, "currency"),
            "Price": round(price, 2) if price else None,
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout_ratio,
            "PE": safe(info, "trailingPE"),
            "DivCAGR_5Y_%": None,   # kommer senere
            "Score": None,          # kommer senere
            "Signal": None,         # kommer senere
            "Confidence": None,     # kommer senere
            "Why": None             # kommer senere
        }

        rows.append(row)

    except Exception as e:
        print(f"❌ Error on {ticker}: {e}")

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(rows, columns=[
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Currency",
    "Price",
    "DividendYield_%",
    "PayoutRatio_%",
    "PE",
    "DivCAGR_5Y_%",
    "Score",
    "Signal",
    "Confidence",
    "Why"
])

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")
