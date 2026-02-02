import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

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

MIN_YEARS_FOR_CAGR = 5

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
    current_year = datetime.now().year
    annual_divs = annual_divs[annual_divs.index < current_year]

    if len(annual_divs) < 2:
        return 0

    years = 0
    values = annual_divs.values

    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            years += 1
        else:
            break

    return years

def calculate_div_cagr_5y(annual_divs: pd.Series) -> float:
    current_year = datetime.now().year
    annual_divs = annual_divs[annual_divs.index < current_year]

    if len(annual_divs) < MIN_YEARS_FOR_CAGR + 1:
        return 0.0

    recent = annual_divs.tail(MIN_YEARS_FOR_CAGR + 1)

    start = recent.iloc[0]
    end = recent.iloc[-1]

    if start <= 0 or end <= 0:
        return 0.0

    cagr = (end / start) ** (1 / MIN_YEARS_FOR_CAGR) - 1
    return round(cagr * 100, 2)

# -------------------------
# MAIN
# -------------------------
rows = []

for ticker in TICKER
