import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# =========================
# SIKKER MAPPE
# =========================
os.makedirs("data", exist_ok=True)

# =========================
# UNIVERS (3A – STORT)
# =========================
TICKERS = [
    # 🇺🇸 USA
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","HD","LOW","WM","COST",
    "ABBV","ABT","UNH","ADP","CAT","EMR","O","V","MA","TXN","AVGO",
    "NEE","DUK","SO","XOM","CVX",

    # 🇸🇪 Sverige
    "ATCO-A.ST","ATCO-B.ST","ASSA-B.ST","VOLV-B.ST","SHB-A.ST","SWED-A.ST",
    "SEB-A.ST","TEL2-B.ST","TELIA.ST","ESSITY-B.ST","SCA-B.ST","SKF-B.ST",
    "INDU-C.ST","EQT.ST","INVE-B.ST",

    # 🇩🇰 Danmark
    "NOVO-B.CO","CARL-B.CO","ORSTED.CO","PNDORA.CO",

    # 🇨🇦 Canada
    "ENB.TO","TRP.TO","BCE.TO","RCI-B.TO","TD.TO","RY.TO","BMO.TO"
]

# =========================
# HJÆLPERE
# =========================
def country_from_ticker(t):
    if t.endswith(".ST"): return "Sweden"
    if t.endswith(".CO"): return "Denmark"
    if t.endswith(".TO"): return "Canada"
    return "United States"

def ltm_dividend(divs):
    if divs is None or divs.empty:
        return 0.0
    cutoff = datetime.utcnow() - timedelta(days=365)
    return divs[divs.index >= cutoff].sum()

def years_growing(divs):
    if divs is None or divs.empty:
        return 0
    yearly = divs.resample("Y").sum()
    count = 0
    for i in range(len(yearly)-1, 0, -1):
        if yearly.iloc[i] > yearly.iloc[i-1] > 0:
            count += 1
        else:
            break
    return count

def cagr_5y(divs):
    if divs is None or divs.empty:
        return 0.0
    yearly = divs.resample("Y").sum()
    if len(yearly) < 6:
        return 0.0
    start = yearly.iloc[-6]
    end = yearly.iloc[-1]
    if start <= 0:
        return 0.0
    return ((end / start) ** (1/5) - 1) * 100

# =========================
# MAIN
# =========================
rows = []

for t in TICKERS:
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        divs = tk.dividends

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or price <= 0:
            continue

        dy = (ltm_dividend(divs) / price) * 100
        payout = info.get("payoutRatio") or 0
        roe = info.get("returnOnEquity") or 0

        yg = years_growing(divs)
        cagr = cagr_5y(divs)

        score = (
            min(dy, 6) * 6 +
            min(roe * 100, 25) +
            min(yg * 2, 20)
        )

        signal = "BUY" if score >= 70 else "HOLD" if score >= 40 else "WATCH"

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName", t),
            "Country": country_from_ticker(t),
            "Sector": info.get("sector", ""),
            "Price": round(price, 2),
            "DividendYield_%": round(dy, 2),
            "PayoutRatio_%": round(payout * 100, 2),
            "ROE_%": round(roe * 100, 2),
            "YearsGrowing": yg,
            "DivCAGR_5Y_%": round(cagr, 2),
            "Score": round(score, 1),
            "Signal": signal
        })

    except Exception as e:
        print(f"Skip {t}: {e}")

df = pd.DataFrame(rows)
df["GeneratedUTC"] = datetime.utcnow().isoformat()

df.to_csv("data/screener_results.csv", index=False)
print(f"Wrote {len(df)} rows")
