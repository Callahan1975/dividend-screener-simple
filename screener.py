import pandas as pd
import yfinance as yf
from pathlib import Path

# --------------------
# CONFIG
# --------------------
OUTPUT_PATH = Path("data/screener_results.csv")

TICKERS = [
    "AAPL","ABBV","ADP","AMGN","ARCC","CL","CVX","D","DNB.OL","DUK","ED","EMR",
    "ENB.TO","EQNR.OL","JNJ","KO","MSFT","O","PEP","PG","RY.TO",
    "BMO.TO","BNS.TO","TD.TO",
    "ASSA-B.ST","ATCO-A.ST","ATCO-B.ST","SEB-A.ST","SHB-A.ST","SWED-A.ST",
    "TEL2-B.ST","TELIA.ST","VOLV-B.ST",
    "CARL-B.CO","NOVO-B.CO","ORSTED.CO",
    "CNQ.TO","COST","XOM","WM"
]

# --------------------
# HELPERS
# --------------------
def safe(v, default=0):
    if v is None or pd.isna(v):
        return default
    return v

def dividend_class(streak):
    if streak >= 50:
        return "King"
    if streak >= 25:
        return "Aristocrat"
    if streak >= 10:
        return "Contender"
    return "None"

# --------------------
# MAIN
# --------------------
rows = []

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("dividendRate"))
        yield_pct = round((dividend / price) * 100, 2) if price > 0 and dividend > 0 else 0

        payout = safe(info.get("payoutRatio")) * 100
        roe = safe(info.get("returnOnEquity")) * 100

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": round(price, 2),
            "DividendYield_pct": round(yield_pct, 2),
            "PayoutRatio_pct": round(payout, 2),
            "ROE_pct": round(roe, 2),
            "YearsGrowing": 0,
            "DividendStreak": 0,
            "DividendClass": "None",
            "Score": 0,
            "Signal": "NONE"
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
