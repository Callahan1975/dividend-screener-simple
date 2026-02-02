import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
TICKERS = [
    "AAPL","ABBV","ADP","AMGN","ARCC",
    "ASSA-B.ST","ATCO-A.ST","ATCO-B.ST",
    "BMO.TO","BNS.TO","CARL-B.CO","CAT","CL",
    "CNQ.TO","COST","CVX","D","DNB.OL","DUK",
    "ED","EMR","ENB.TO","EQNR.OL","JNJ","KO",
    "MSFT","NOVO-B.CO","O","ORSTED.CO","PEP",
    "PG","RY.TO","SEB-A.ST","SHB-A.ST","SWED-A.ST",
    "TD.TO","TEL2-B.ST","TELIA.ST","TRP.TO",
    "VOLV-B.ST","WM","XOM"
]

DIVIDEND_HISTORY_PATH = "data/dividend_history/dividend_history.csv"

OUTPUT_PATHS = [
    "data/screener_results/screener_results.csv",
    "docs/data/screener_results/screener_results.csv"
]

# =========================
# LOAD DIVIDEND HISTORY
# =========================
if os.path.exists(DIVIDEND_HISTORY_PATH):
    div_hist = pd.read_csv(DIVIDEND_HISTORY_PATH)
else:
    div_hist = pd.DataFrame(columns=["Ticker", "Year", "Dividend"])

# =========================
# DIVIDEND METRICS
# =========================
def years_growing(ticker):
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 2:
        return 0

    values = df["Dividend"].values
    count = 0

    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break

    return count

def div_cagr_5y(ticker):
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 6:
        return 0.0

    recent = df.tail(6)
    start = recent.iloc[0]["Dividend"]
    end = recent.iloc[-1]["Dividend"]

    if start <= 0 or end <= 0:
        return 0.0

    return round(((end / start) ** (1 / 5) - 1) * 100, 2)

# =========================
# MAIN
# =========================
rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": info.get("currentPrice", 0),
            "DividendYield_%": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio", 0) or 0) * 100, 2),
            "ROE_%": round((info.get("returnOnEquity", 0) or 0) * 100, 2),
            "YearsGrowing": years_growing(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),
            "Score": 0,
            "Signal": "WATCH"
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

df = pd.DataFrame(rows)

# =========================
# SAVE (CI SAFE)
# =========================
for path in OUTPUT_PATHS:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

print("Hybrid model complete – dividend history loaded from CSV")
