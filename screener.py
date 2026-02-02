# screener.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# 1. UNIVERS (STORT + SE)
# =========================
TICKERS = [
    # USA
    "AAPL","MSFT","GOOGL","META","AMZN","NVDA","AVGO",
    "JNJ","PG","KO","PEP","CL","KMB",
    "O","VICI","PLD",
    "XOM","CVX","COP",
    "NEE","DUK","SO",
    "JPM","BAC","WFC","V","MA","UNH","HD","WM",

    # Sverige
    "ATCO-A.ST","ATCO-B.ST","VOLV-A.ST","VOLV-B.ST",
    "ASSA-B.ST","INVE-A.ST","INVE-B.ST",
    "SEB-A.ST","SHB-A.ST","SWED-A.ST",
    "TEL2-B.ST","TELIA.ST",
    "SCA-B.ST","ESSITY-B.ST","SKF-B.ST",
    "EQT.ST","INDU-C.ST","LATO-B.ST",

    # Danmark
    "NOVO-B.CO","CARL-B.CO","ORSTED.CO","PNDORA.CO","DSV.CO",

    # Canada
    "RY.TO","TD.TO","BMO.TO","BNS.TO",
    "ENB.TO","TRP.TO","CNQ.TO",

    # Norge
    "DNB.OL","ORK.OL","YAR.OL"
]

# =========================
# 2. HJÆLPEFUNKTIONER
# =========================
def country_from_ticker(t):
    if t.endswith(".ST"): return "Sweden"
    if t.endswith(".CO"): return "Denmark"
    if t.endswith(".TO"): return "Canada"
    if t.endswith(".OL"): return "Norway"
    return "United States"

def calc_ltm_dividend(divs: pd.Series) -> float:
    if divs is None or divs.empty:
        return 0.0
    cutoff = datetime.utcnow() - timedelta(days=365)
    return divs[divs.index >= cutoff].sum()

def calc_years_growing(divs: pd.Series) -> int:
    if divs is None or divs.empty:
        return 0
    yearly = divs.resample("Y").sum()
    years = 0
    for i in range(len(yearly)-1, 0, -1):
        if yearly.iloc[i] > yearly.iloc[i-1] > 0:
            years += 1
        else:
            break
    return years

def calc_cagr_5y(divs: pd.Series) -> float:
    if divs is None or divs.empty:
        return 0.0
    yearly = divs.resample("Y").sum()
    if len(yearly) < 6:
        return 0.0
    start = yearly.iloc[-6]
    end = yearly.iloc[-1]
    if start <= 0:
        return 0.0
    return (end / start) ** (1/5) - 1

# =========================
# 3. DATAINDSAMLING
# =========================
rows = []

for t in TICKERS:
    try:
        yf_t = yf.Ticker(t)
        info = yf_t.info or {}
        divs = yf_t.dividends

        price = info.get("regularMarketPrice") or info.get("currentPrice") or np.nan

        ltm_div = calc_ltm_dividend(divs)
        div_yield = (ltm_div / price * 100) if price and price > 0 else 0.0

        payout = info.get("payoutRatio")
        payout = payout * 100 if isinstance(payout, (int, float)) else 0.0

        roe = info.get("returnOnEquity")
        roe = roe * 100 if isinstance(roe, (int, float)) else 0.0

        years_growing = calc_years_growing(divs)
        div_cagr = calc_cagr_5y(divs) * 100

        score = (
            min(div_yield, 6) * 6 +
            min(roe, 25) * 1.2 +
            min(years_growing, 25) * 1.0
        )

        signal = "BUY" if score >= 70 else "HOLD" if score >= 40 else "WATCH"

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName", t),
            "Country": country_from_ticker(t),
            "Sector": info.get("sector", "Unknown"),
            "Price": round(price, 2) if price else 0,
            "DividendYield_%": round(div_yield, 2),
            "PayoutRatio_%": round(payout, 2),
            "ROE_%": round(roe, 2),
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": round(div_cagr, 2),
            "Score": round(score, 1),
            "Signal": signal
        })

    except Exception as e:
        print(f"Skip {t}: {e}")

# =========================
# 4. CSV OUTPUT (ALTID!)
# =========================
df = pd.DataFrame(rows)

if df.empty:
    df = pd.DataFrame([{
        "Ticker": "INFO",
        "Name": "No data",
        "Country": "",
        "Sector": "",
        "Price": 0,
        "DividendYield_%": 0,
        "PayoutRatio_%": 0,
        "ROE_%": 0,
        "YearsGrowing": 0,
        "DivCAGR_5Y_%": 0,
        "Score": 0,
        "Signal": "WATCH"
    }])

df.to_csv("screener_results.csv", index=False)
df.to_csv("docs/data/screener_results.csv", index=False)

print(f"Generated {len(df)} rows")
