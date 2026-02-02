import yfinance as yf
import pandas as pd
from datetime import datetime

TICKERS = [
    # USA
    "AAPL","MSFT","JNJ","PG","KO","PEP","ADP","ABBV","AMGN","CAT","CVX","XOM",
    "COST","CL","WM","EMR","ED","DUK","D","NEE","O","ARCC","V","MA",

    # Sweden
    "ASSA-B.ST","ATCO-A.ST","ATCO-B.ST","INVE-B.ST","VOLV-B.ST","SEB-A.ST",
    "SHB-A.ST","SWED-A.ST","TEL2-B.ST","TELIA.ST","EQT.ST",

    # Denmark
    "CARL-B.CO","NOVO-B.CO","ORSTED.CO",

    # Canada
    "ENB.TO","BMO.TO","BNS.TO","TD.TO","RY.TO","TRP.TO","CNQ.TO",

    # Norway
    "DNB.OL"
]

rows = []

for t in TICKERS:
    try:
        tk = yf.Ticker(t)
        info = tk.info

        price = info.get("currentPrice") or 0
        dividend = info.get("dividendRate") or 0

        yield_pct = round((dividend / price) * 100, 2) if price > 0 else 0

        payout = info.get("payoutRatio")
        payout = round(payout * 100, 2) if payout is not None else 0

        roe = info.get("returnOnEquity")
        roe = round(roe * 100, 2) if roe is not None else 0

        score = 0
        score += min(yield_pct, 5) * 5
        score += min(roe, 20)
        score += max(0, 10 - abs(payout - 50) / 5)

        if score >= 60:
            signal = "BUY"
        elif score >= 40:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName",""),
            "Country": info.get("country",""),
            "Sector": info.get("sector",""),
            "Price": round(price,2),
            "DividendYield_%": yield_pct,
            "PayoutRatio_%": payout,
            "ROE_%": roe,
            "YearsGrowing": 0,
            "DivCAGR_5Y_%": 0,
            "Score": round(score,1),
            "Signal": signal
        })

    except Exception as e:
        continue

df = pd.DataFrame(rows)
df["GeneratedUTC"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

df.to_csv("data/screener_results.csv", index=False)
print(f"Saved {len(df)} rows")
