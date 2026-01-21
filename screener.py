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

        # ---- rate-limit protection ----
        time.sleep(2)

        info = t.info
        if not info:
            print(f"SKIP {ticker}: no info")
            continue

        price = safe_float(info.get("currentPrice"))
        dividend = safe_float(info.get("dividendRate"))
        eps = safe_float(info.get("trailingEps"))
        pe = safe_float(info.get("trailingPE"))

        # Price is mandatory
        if not price:
            print(f"SKIP {ticker}: no price")
            continue

        # Dividend Yield (%)
        dividend_yield = None
        if dividend and price:
            dividend_yield = round((dividend / price) * 100, 2)

        # Payout Ratio (%)
        payout_ratio = None
        if dividend and eps and eps > 0:
            payout_ratio = round((dividend / eps) * 100, 1)

        rows.append({
            "GeneratedUTC": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Currency": info.get("currency"),
            "Exchange": info.get("exchange"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Price": round(price, 2),
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout_ratio,
            "PE": round(pe, 1) if pe else None
        })

        print(f"OK {ticker}")

    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        time.sleep(5)

# -------------------------------------------------
# SAFE WRITE CSV (ALWAYS)
# -------------------------------------------------
if rows:
    df = pd.DataFrame(rows)
else:
    # Create EMPTY dataframe with correct columns
    df = pd.DataFrame(columns=COLUMNS)

df = df.reindex(columns=COLUMNS)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✔ Screener finished – {len(df)} rows written")
