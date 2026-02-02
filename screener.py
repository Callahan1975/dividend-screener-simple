import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

TICKERS = [
    # US
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","HD","V","MA","UNH","ALL","TRV","VZ","T","O","WM",
    # Canada
    "BCE.TO","RCI-B.TO","ENB.TO","BMO.TO","TD.TO",
    # Nordic
    "NOVO-B.CO","PNDORA.CO","ORSTED.CO","ORK.OL","DNB.OL"
]

def dividend_years_and_cagr(divs):
    if divs.empty:
        return 0, 0.0

    yearly = divs.resample("Y").sum()
    yearly = yearly[yearly > 0]

    years = len(yearly)
    if years < 2:
        return years, 0.0

    start = yearly.iloc[0]
    end = yearly.iloc[-1]
    cagr = (end / start) ** (1 / (years - 1)) - 1
    return years, round(cagr * 100, 2)

rows = []

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            continue

        divs = stock.dividends
        years_growing, div_cagr = dividend_years_and_cagr(divs)

        dividend_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
        payout = info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0
        roe = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0

        score = (
            min(years_growing * 5, 40) +
            min(div_cagr * 2, 30) +
            min(roe, 20) -
            min(max(payout - 75, 0), 20)
        )

        if years_growing >= 10 and div_cagr >= 5 and payout <= 75 and roe >= 15:
            signal = "GOLD"
        elif years_growing >= 5 and div_cagr >= 3:
            signal = "BUY"
        elif years_growing >= 2:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": round(price, 2),
            "DividendYield_%": round(dividend_yield, 2),
            "PayoutRatio_%": round(payout, 2),
            "ROE_%": round(roe, 2),
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": div_cagr,
            "Score": round(score, 1),
            "Signal": signal,
            "GeneratedUTC": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print(f"Error {ticker}: {e}")

df = pd.DataFrame(rows)

if df.empty:
    df = pd.DataFrame([{
        "Ticker": "NO_DATA",
        "Name": "No data generated",
        "GeneratedUTC": datetime.utcnow().isoformat()
    }])

df.to_csv("data/screener_results.csv", index=False)
print("✅ screener_results.csv generated")
