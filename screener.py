import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------
# CONFIG
# -------------------------
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

MIN_YEARS_CAGR = 5

# -------------------------
# HELPERS
# -------------------------
def annualize_dividends(dividends: pd.Series) -> pd.Series:
    if dividends.empty:
        return pd.Series(dtype=float)

    df = dividends.to_frame(name="div")
    df["year"] = df.index.year
    annual = df.groupby("year")["div"].sum()
    return annual.sort_index()

def calculate_years_growing(annual_divs: pd.Series) -> int:
    if len(annual_divs) < 2:
        return 0

    # Exclude current year
    current_year = datetime.now().year
    annual_divs = annual_divs[annual_divs.index < current_year]

    if len(annual_divs) < 2:
        return 0

    years = 0
    values = annual_divs.values

    for i in range(len(values)-1, 0, -1):
        if values[i] > values[i-1]:
            years += 1
        else:
            break

    return years

def calculate_div_cagr(annual_divs: pd.Series, years: int) -> float:
    if years < MIN_YEARS_CAGR:
        return 0.0

    recent = annual_divs.tail(years + 1)
    if len(recent) < years + 1:
        return 0.0

    start = recent.iloc[0]
    end = recent.iloc[-1]

    if start <= 0 or end <= 0:
        return 0.0

    return round(((end / start) ** (1 / years) - 1) * 100, 2)

# -------------------------
# MAIN
# -------------------------
rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        dividends = stock.dividends

        annual_divs = annualize_dividends(dividends)
        years_growing = calculate_years_growing(annual_divs)
        div_cagr_5y = calculate_div_cagr(annual_divs, 5)

        row = {
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": info.get("currentPrice", 0),
            "DividendYield_%": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio", 0) or 0) * 100, 2),
            "ROE_%": round((info.get("returnOnEquity", 0) or 0) * 100, 2),
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": div_cagr_5y,
            "Score": 0,
            "Signal": "WATCH"
        }

        rows.append(row)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

df = pd.DataFrame(rows)

# -------------------------
# SAVE
# -------------------------
for path in OUTPUT_PATHS:
    df.to_csv(path, index=False)

print("Fase 2A complete – dividend history calculated.")
