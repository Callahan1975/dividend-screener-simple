import yfinance as yf
import pandas as pd
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

OUTPUT_PATHS = [
    "data/screener_results/screener_results.csv",
    "docs/data/screener_results/screener_results.csv"
]

# =========================
# DIVIDEND HELPERS
# =========================
def get_annual_dividends(ticker):
    hist = yf.Ticker(ticker).history(period="max", actions=True)

    if "Dividends" not in hist or hist["Dividends"].sum() == 0:
        return pd.Series(dtype=float)

    df = hist[hist["Dividends"] > 0][["Dividends"]].copy()
    df["Year"] = df.index.year

    return df.groupby("Year")["Dividends"].sum().sort_index()

def years_growing(annual):
    current_year = datetime.now().year
    annual = annual[annual.index < current_year]

    if len(annual) < 2:
        return 0

    values = annual.values
    count = 0

    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break

    return count

def div_cagr_5y(annual):
    current_year = datetime.now().year
    annual = annual[annual.index < current_year]

    if len(annual) < 6:
        return 0.0

    recent = annual.tail(6)
    start, end = recent.iloc[0], recent.iloc[-1]

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

        annual_divs = get_annual_dividends(ticker)

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": info.get("currentPrice", 0),
            "DividendYield_%": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio", 0) or 0) * 100, 2),
            "ROE_%": round((info.get("returnOnEquity", 0) or 0) * 100, 2),
            "YearsGrowing": years_growing(annual_divs),
            "DivCAGR_5Y_%": div_cagr_5y(annual_divs),
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

print("Fase 2A complete – dividends loaded via history()")
