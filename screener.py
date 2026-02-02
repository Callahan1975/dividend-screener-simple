import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# =========================
# TICKER UNIVERSE
# =========================
TICKERS = [
    # USA
    "AAPL","MSFT","JNJ","PG","KO","PEP","ADP","ABBV","AMGN","CAT","CL","COST",
    "CVX","XOM","D","DUK","ED","EMR","ALL","BA","ABT",
    # Canada
    "BMO.TO","BNS.TO","RY.TO","TD.TO","ENB.TO","BCE.TO",
    # Sweden
    "ATCO-A.ST","ATCO-B.ST","ASSA-B.ST","VOLV-B.ST","SEB-A.ST","SHB-A.ST",
    "SWED-A.ST","TEL2-B.ST","TELIA.ST","EQT.ST","INVE-B.ST",
    # Denmark
    "CARL-B.CO","NOVO-B.CO","DSV.CO","TRYG.CO",
    # Norway
    "DNB.OL"
]

# =========================
# HELPERS
# =========================
def safe(v):
    if v is None or pd.isna(v):
        return 0.0
    return float(v)

def calc_dividend_yield(dividends, price):
    if dividends is None or dividends.empty or price <= 0:
        return 0.0
    last_12m = dividends[dividends.index >= (datetime.now() - timedelta(days=365))]
    total = last_12m.sum()
    return round((total / price) * 100, 2)

def calc_years_growing(dividends):
    if dividends is None or dividends.empty:
        return 0
    yearly = dividends.resample("Y").sum()
    count = 0
    for i in range(len(yearly) - 1, 0, -1):
        if yearly.iloc[i] > yearly.iloc[i - 1]:
            count += 1
        else:
            break
    return count

def calc_div_cagr(dividends, years=5):
    if dividends is None or dividends.empty:
        return 0.0
    yearly = dividends.resample("Y").sum()
    if len(yearly) < years + 1:
        return 0.0
    start = yearly.iloc[-(years + 1)]
    end = yearly.iloc[-1]
    if start <= 0:
        return 0.0
    return round(((end / start) ** (1 / years) - 1) * 100, 2)

# =========================
# MAIN LOOP
# =========================
rows = []

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))

        dividends = t.dividends

        dividend_yield = calc_dividend_yield(dividends, price)
        payout = safe(info.get("payoutRatio")) * 100
        roe = safe(info.get("returnOnEquity")) * 100

        years_growing = calc_years_growing(dividends)
        div_cagr = calc_div_cagr(dividends)

        score = (
            min(dividend_yield, 6) * 5 +
            min(div_cagr, 10) * 2 +
            min(roe, 30) +
            min(years_growing * 2, 20)
        )

        if score >= 70:
            signal = "BUY"
        elif score >= 40:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": round(price, 2),
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": round(payout, 2),
            "ROE_%": round(roe, 2),
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": div_cagr,
            "Score": round(score, 1),
            "Signal": signal
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

# =========================
# OUTPUT
# =========================
df = pd.DataFrame(rows)

df.to_csv("data/screener_results.csv", index=False)
print(f"Saved {len(df)} rows at {datetime.utcnow()} UTC")
