import yfinance as yf
import pandas as pd
from datetime import datetime
import math

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "data/screener_results.csv"

TICKERS = [
    # 🇺🇸 USA
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","WMT","COST","HD","LOW",
    "UNH","VZ","T","ALL","TRV","CVX","XOM","IBM","ADP","MMM","CL","KMB",
    "ABT","ABBV","TXN","QCOM","AMGN","MDT","UPS","CAT","EMR","ETN",
    "HON","RTX","LMT","NOC","GD","BA","NEE","DUK","SO","D","ED","SRE",
    "XEL","PEG","WEC","DTE","O","VICI",

    # 🇸🇪 Sverige
    "ATCO-A.ST","ATCO-B.ST","VOLV-B.ST","SHB-A.ST","SEB-A.ST",
    "INVE-B.ST","TEL2-B.ST","TELIA.ST","HM-B.ST","ERIC-B.ST",
    "EQT.ST","ASSA-B.ST","SCA-B.ST","SKF-B.ST","SAND.ST",

    # 🇩🇰 Danmark
    "NOVO-B.CO","PNDORA.CO","MAERSK-B.CO","ORSTED.CO","CARL-B.CO",

    # 🇨🇦 Canada
    "ENB.TO","BCE.TO","RCI-B.TO","TRP.TO","BMO.TO","TD.TO","RY.TO",

    # 🇳🇴 Norge
    "DNB.OL","ORK.OL","YAR.OL"
]

# -----------------------------
# HELPERS
# -----------------------------
def safe(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

def dividend_stats(ticker):
    try:
        divs = ticker.dividends
        if divs is None or len(divs) < 2:
            return 0, 0.0

        yearly = divs.resample("Y").sum()
        years = yearly[yearly > 0]

        years_growing = 0
        for i in range(1, len(years)):
            if years.iloc[i] > years.iloc[i-1]:
                years_growing += 1
            else:
                break

        if len(years) >= 6:
            start = years.iloc[-6]
            end = years.iloc[-1]
            if start > 0:
                cagr = (end / start) ** (1/5) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        return years_growing, round(cagr * 100, 2)

    except Exception:
        return 0, 0.0

# -----------------------------
# MAIN
# -----------------------------
rows = []

for symbol in TICKERS:
    try:
        t = yf.Ticker(symbol)
        info = t.info

        price = safe(info.get("currentPrice"))
        if price is None:
            continue  # drop only truly dead tickers

        dividend_yield = safe(info.get("dividendYield"))
        dividend_yield = round(dividend_yield * 100, 2) if dividend_yield else 0.0

        payout = safe(info.get("payoutRatio"))
        payout = round(payout * 100, 2) if payout else 0.0

        roe = safe(info.get("returnOnEquity"))
        roe = round(roe * 100, 2) if roe else 0.0

        years_growing, div_cagr = dividend_stats(t)

        score = (
            min(dividend_yield * 2, 20)
            + min(roe / 2, 20)
            + min(years_growing * 2, 20)
            + min(div_cagr, 20)
        )

        if score >= 60:
            signal = "BUY"
        elif score >= 40:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "Ticker": symbol,
            "Name": info.get("shortName", symbol),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": round(price, 2),
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout,
            "ROE_%": roe,
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": div_cagr,
            "Score": round(score, 1),
            "Signal": signal
        })

    except Exception:
        continue

# -----------------------------
# WRITE CSV
# -----------------------------
df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No data generated – check ticker universe")

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
