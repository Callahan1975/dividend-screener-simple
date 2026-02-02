import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
TICKERS = [
    "AAPL","ABBV","ADP","AMGN","ARCC","PG","KO","JNJ","PEP","EMR","CL",
    "ED","D","WM","DUK","MSFT","COST","CVX","XOM",
    "BMO.TO","BNS.TO","RY.TO","TD.TO","ENB.TO","CNQ.TO","TRP.TO",
    "CARL-B.CO","NOVO-B.CO","ORSTED.CO",
    "SEB-A.ST","SHB-A.ST","SWED-A.ST","TELIA.ST","TEL2-B.ST",
    "VOLV-B.ST","ASSA-B.ST","ATCO-A.ST","ATCO-B.ST",
    "EQNR.OL","DNB.OL"
]

OUTPUT_PATH = "data/screener_results.csv"

# -------------------------
# HELPERS
# -------------------------
def safe(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 0.0
    return v

# -------------------------
# MAIN
# -------------------------
rows = []

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("trailingAnnualDividendRate"))

        dividend_yield = (dividend / price * 100) if price > 0 else 0

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName",""),
            "Country": info.get("country",""),
            "Sector": info.get("sector",""),
            "Price": round(price,2),
            "DividendYield_%": round(dividend_yield,2),
            "PayoutRatio_%": round(safe(info.get("payoutRatio"))*100,2),
            "ROE_%": round(safe(info.get("returnOnEquity"))*100,2),
            "YearsGrowing": 0,
            "DivCAGR_5Y_%": 0,
            "DividendStreak": 0,
            "DividendClass": "",
            "Score": 0,
            "Signal": ""
        })

    except Exception as e:
        print("Error:", ticker, e)

df = pd.DataFrame(rows)

# -------- CLEAN (CRITICAL FOR DATATABLES) --------
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

# -------------------------
# SAVE
# -------------------------
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
