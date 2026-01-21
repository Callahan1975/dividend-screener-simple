import yfinance as yf
import pandas as pd
import time
from datetime import datetime

TICKERS = [
    "AAPL","ABBV","ABT","ADP","AEP","AFL","AIG","AMCR","AMGN","AOS",
    "APD","AVGO","AWK","BAC","BDX","BLK","CARL-B.CO","CAT"
]

ROWS = []

def safe(v):
    return None if v in [None, "", "nan"] else v

for t in TICKERS:
    print(f"Fetching {t}")
    try:
        tk = yf.Ticker(t)
        info = tk.info

        price = safe(info.get("currentPrice"))
        dividend_yield = None

        # 1) Preferred: dividendYield
        if info.get("dividendYield"):
            dividend_yield = round(info["dividendYield"] * 100, 2)

        # 2) Fallback: dividendRate / price
        elif info.get("dividendRate") and price:
            dividend_yield = round((info["dividendRate"] / price) * 100, 2)

        # 3) LAST fallback (Apple-safe)
        elif info.get("trailingAnnualDividendRate") and price:
            dividend_yield = round(
                info["trailingAnnualDividendRate"] / price * 100, 2
            )

        payout = safe(info.get("payoutRatio"))
        if payout is not None:
            payout = round(payout * 100, 1)

        ROWS.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": t,
            "Name": info.get("longName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Price": price,
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout,
            "PE": info.get("trailingPE")
        })

        time.sleep(2)  # 🚨 RATE LIMIT FIX

    except Exception as e:
        print(f"ERROR {t}: {e}")

df = pd.DataFrame(ROWS)

# 🔒 GARANTÉR KOLONNER (ALDRIG KeyError)
EXPECTED_COLS = [
    "GeneratedUTC","Ticker","Name","Country","Sector",
    "Price","DividendYield_%","PayoutRatio_%","PE"
]

for c in EXPECTED_COLS:
    if c not in df.columns:
        df[c] = None

df = df[EXPECTED_COLS]

df.to_csv("data/screener_results.csv", index=False)
print("DONE")
