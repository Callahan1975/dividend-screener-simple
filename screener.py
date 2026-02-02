import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# UNIVERSE (kan udvides)
# =========================
TICKERS = [
    # USA
    "AAPL","MSFT","KO","PG","PEP","JNJ","ABBV","CVX","XOM","O","ARCC",
    "COST","WM","CAT","ADP","AMGN","CL","D","DUK","ED","EMR",
    # Canada
    "RY.TO","TD.TO","BMO.TO","BNS.TO","ENB.TO","CNQ.TO","TRP.TO",
    # Sweden
    "ATCO-A.ST","ATCO-B.ST","ASSA-B.ST","VOLV-B.ST","SHB-A.ST",
    "SEB-A.ST","SWED-A.ST","TEL2-B.ST","TELIA.ST","EQNR.OL",
    # Denmark / Nordics
    "NOVO-B.CO","CARL-B.CO","ORSTED.CO","DNB.OL"
]

# =========================
# HELPERS
# =========================
def safe(v):
    return None if v in [None, "None"] else v

rows = []

for t in TICKERS:
    try:
        tk = yf.Ticker(t)
        info = tk.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            continue

        dividend = info.get("dividendRate") or 0
        yield_pct = round((dividend / price) * 100, 2) if dividend else 0

        payout = info.get("payoutRatio")
        roe = info.get("returnOnEquity")

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Price": round(price, 2),
            "DividendYield_%": yield_pct,
            "PayoutRatio_%": round(payout * 100, 2) if payout else None,
            "ROE_%": round(roe * 100, 2) if roe else None,
            "YearsGrowing": 0,            # klar til fase 3b
            "DivCAGR_5Y_%": 0,             # klar til fase 3b
        })

    except Exception as e:
        print(f"Skip {t}: {e}")

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No data collected")

# =========================
# SCORE + SIGNAL
# =========================
df["Score"] = (
    df["DividendYield_%"].fillna(0) * 6 +
    df["ROE_%"].fillna(0) * 0.2
).round(1)

def signal(row):
    if row["Score"] >= 45:
        return "BUY"
    if row["Score"] >= 35:
        return "HOLD"
    return "WATCH"

df["Signal"] = df.apply(signal, axis=1)

df["GeneratedUTC"] = datetime.utcnow().isoformat()

# =========================
# SAVE
# =========================
Path("data").mkdir(exist_ok=True)
df.to_csv("data/screener_results.csv", index=False)

print(f"Saved {len(df)} rows")
