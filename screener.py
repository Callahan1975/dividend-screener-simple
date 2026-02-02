import yfinance as yf
import pandas as pd
from datetime import datetime

TICKERS = [
    # 🇺🇸 USA
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","HD","LOW",
    "WM","COST","ABBV","ABT","UNH","ADP","CAT","EMR",
    "O","V","MA","TXN","AVGO","NEE","DUK","SO",

    # 🇸🇪 Sweden
    "ATCO-A.ST","ATCO-B.ST","ASSA-B.ST","VOLV-B.ST",
    "SHB-A.ST","SWED-A.ST","SEB-A.ST","TEL2-B.ST","TELIA.ST",
    "ESSITY-B.ST","SCA-B.ST","SKF-B.ST","INDU-C.ST","EQT.ST",

    # 🇩🇰 Denmark
    "NOVO-B.CO","CARL-B.CO","ORSTED.CO","PNDORA.CO",

    # 🇨🇦 Canada
    "ENB.TO","TRP.TO","BCE.TO","RCI-B.TO","TD.TO","RY.TO","BMO.TO"
]

rows = []

for t in TICKERS:
    try:
        tk = yf.Ticker(t)
        info = tk.info

        price = info.get("currentPrice")
        if price is None:
            continue

        dividend = info.get("dividendRate") or 0
        dividend_yield = round((dividend / price) * 100, 2) if dividend > 0 else 0

        payout = info.get("payoutRatio") or 0
        roe = info.get("returnOnEquity") or 0

        score = round(
            dividend_yield * 5 +
            (roe * 100) -
            (payout * 10),
            1
        )

        signal = "BUY" if score >= 45 else "HOLD" if score >= 35 else "WATCH"

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName",""),
            "Country": info.get("country",""),
            "Sector": info.get("sector",""),
            "Price": round(price,2),
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": round(payout * 100,2),
            "ROE_%": round(roe * 100,2),
            "YearsGrowing": 0,
            "DivCAGR_5Y_%": 0,
            "Score": score,
            "Signal": signal
        })

    except Exception as e:
        print(f"Skip {t}: {e}")

df = pd.DataFrame(rows)

df.to_csv("screener_results.csv", index=False)
print(f"Saved {len(df)} rows @ {datetime.utcnow()} UTC")
