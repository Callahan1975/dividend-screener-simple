import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_TICKERS = [
    "AAPL","ABBV","ABT","ADP","AEP","AFL","AIG","AMCR","AMGN","AOS",
    "APD","AVGO","AWK","BAC","BDX","BLK",
    "NOVO-B.CO","DSV.CO","MAERSK-B.CO","CARL-B.CO","PNDORA.CO",
    "COLO-B.CO","DANSKE.CO","SYDB.CO"
]

OUTPUT_FILE = "data/screener_results.csv"

COLUMNS = [
    "GeneratedUTC","Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Price","DividendYield_%","PayoutRatio_%","PE"
]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return None

# -------------------------------------------------
# MAIN
# -------------------------------------------------
rows = []

for ticker in INPUT_TICKERS:
    try:
        t = yf.Ticker(ticker)
        time.sleep(2)  # rate-limit protection

        info = t.info
        if not info:
            continue

        price = safe_float(info.get("currentPrice"))
        eps = safe_float(info.get("trailingEps"))
        pe = safe_float(info.get("trailingPE"))

        # ---- DIVIDEND LOGIC (ROBUST) ----
        dividend_yield = None

        # 1) Prefer dividendYield (already % in decimal)
        dy = safe_float(info.get("dividendYield"))
        if dy:
            dividend_yield = round(dy * 100, 2)

        # 2) Fallback: dividendRate / price
        else:
            dividend_rate = safe_float(info.get("dividendRate"))
            if dividend_rate and price:
                dividend_yield = round((dividend_rate / price) * 100, 2)

        # ---- PAYOUT RATIO ----
        payout_ratio = None
        dividend_rate = safe_float(info.get("dividendRate"))
        if dividend_rate and eps and eps > 0:
            payout_ratio = round((dividend_rate / eps) * 100, 1)

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Currency": info.get("currency"),
            "Exchange": info.get("exchange"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": round(price, 2) if price else None,
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout_ratio,
            "PE": round(pe, 1) if pe else None
        })

        print(f"OK {ticker}")

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        time.sleep(5)

# -------------------------------------------------
# SAFE WRITE
# -------------------------------------------------
df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✔ Screener finished – {len(df)} rows")
