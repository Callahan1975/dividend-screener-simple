import yfinance as yf
import pandas as pd
from datetime import datetime

# ======================
# UNIVERSE (FASE 3A)
# ======================
TICKERS = [
    # ---------- USA ----------
    "AAPL","MSFT","JNJ","PG","KO","PEP","CL","KMB","WM","HD","LOW","COST",
    "CVX","XOM","V","MA","ABBV","ABT","ADP","NEE","DUK","SO","D","ED","EMR",
    "O","WPC","SPG","VICI","AVGO","TXN","UNH","TROW","MCD","MMM","CAT",
    # ---------- CANADA ----------
    "RY.TO","TD.TO","BMO.TO","BNS.TO","CM.TO","ENB.TO","TRP.TO","FTS.TO",
    # ---------- SWEDEN ----------
    "SHB-A.ST","SEB-A.ST","SWED-A.ST","NDA-SE.ST","ATCO-A.ST","ATCO-B.ST",
    "VOLV-B.ST","ASSA-B.ST","TEL2-B.ST","TELIA.ST","EQT.ST",
    # ---------- DENMARK ----------
    "NOVO-B.CO","ORSTED.CO","CARL-B.CO","PNDORA.CO"
]

# ======================
# HELPERS
# ======================
def get_ttm_dividend(divs: pd.Series) -> float:
    if divs is None or divs.empty:
        return float("nan")
    last_year = divs.index.max()
    one_year_ago = last_year - pd.Timedelta(days=365)
    return divs[divs.index >= one_year_ago].sum()

def calc_years_growing(divs: pd.Series) -> int:
    if divs is None or divs.empty:
        return float("nan")
    yearly = divs.resample("Y").sum()
    yearly = yearly[yearly > 0]
    if len(yearly) < 2:
        return 0
    growth = 0
    for i in range(len(yearly)-1, 0, -1):
        if yearly.iloc[i] > yearly.iloc[i-1]:
            growth += 1
        else:
            break
    return growth

def calc_cagr_5y(divs: pd.Series) -> float:
    if divs is None or divs.empty:
        return float("nan")
    yearly = divs.resample("Y").sum()
    yearly = yearly[yearly > 0]
    if len(yearly) < 6:
        return float("nan")
    start = yearly.iloc[-6]
    end = yearly.iloc[-1]
    if start <= 0:
        return float("nan")
    return ((end / start) ** (1/5) - 1) * 100

# ======================
# MAIN
# ======================
rows = []

for t in TICKERS:
    try:
        y = yf.Ticker(t)
        info = y.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        divs = y.dividends

        ttm_div = get_ttm_dividend(divs)
        div_yield = (ttm_div / price * 100) if price and ttm_div else float("nan")

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Price": price,
            "DividendYield_%": round(div_yield, 2) if pd.notna(div_yield) else float("nan"),
            "PayoutRatio_%": info.get("payoutRatio") * 100 if info.get("payoutRatio") else float("nan"),
            "ROE_%": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else float("nan"),
            "YearsGrowing": calc_years_growing(divs),
            "DivCAGR_5Y_%": round(calc_cagr_5y(divs), 2),
            "Score": float("nan"),
            "Signal": "WATCH"
        })
    except Exception:
        continue

df = pd.DataFrame(rows)
df["GeneratedUTC"] = datetime.utcnow().isoformat()

# Drop helt tomme rækker
df = df.dropna(subset=["Price"], how="all")

# Output (ROOT + /data for Pages)
df.to_csv("screener_results.csv", index=False)
df.to_csv("data/screener_results.csv", index=False)

print(f"Generated {len(df)} rows")
