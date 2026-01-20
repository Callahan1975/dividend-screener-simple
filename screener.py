 # Dividend Screener – Decision Engine (STABLE)
# Outputs CSV used by docs/index.html

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import numpy as np

# ======================
# PATHS
# ======================
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = DATA / "screener_results"
OUT.mkdir(parents=True, exist_ok=True)

ALIAS_FILE = DATA / "ticker_alias.csv"
OUTPUT_FILE = OUT / "screener_results.csv"

# ======================
# LOAD TICKERS
# ======================
alias = pd.read_csv(ALIAS_FILE)
alias["Ticker"] = alias["Ticker"].str.upper().str.strip()
tickers = alias["Ticker"].dropna().unique().tolist()

print(f"Universe size: {len(tickers)}")

# ======================
# CONSTANT LISTS
# ======================
DIVIDEND_KINGS = {
    "PG","KO","JNJ","PEP","EMR","MMM","LOW","CL","KMB","ABT"
}

DIVIDEND_ARISTOCRATS = {
    "AFL","ADP","ABBV","APD","CB","ECL","ITW","MCD","MSFT","TROW"
}

# ======================
# HELPERS
# ======================
def safe(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except:
        return None

def div_cagr_5y(series):
    if series is None or len(series) < 6:
        return None
    start = series.iloc[-6]
    end = series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1/5) - 1) * 100, 2)

# ======================
# MAIN LOOP
# ======================
rows = []

for t in tickers:
    try:
        y = yf.Ticker(t)
        info = y.info

        price = safe(info.get("currentPrice"))
        dividend = safe(info.get("dividendRate"))
        yield_pct = round((dividend / price) * 100, 2) if dividend and price else None
        payout = safe(info.get("payoutRatio"))
        pe = safe(info.get("trailingPE"))

        hist_div = y.dividends.resample("Y").sum()
        cagr = div_cagr_5y(hist_div)

        # ======================
        # SCORE MODEL (0–100)
        # ======================
        score = 0

        if yield_pct:
            score += min(yield_pct * 5, 25)

        if cagr:
            score += min(cagr * 2, 30)

        if payout:
            if payout < 0.75:
                score += 20
            elif payout < 1.0:
                score += 10
            else:
                score -= 20

        if pe:
            if pe < 18:
                score += 15
            elif pe < 25:
                score += 5
            else:
                score -= 10

        score = int(max(min(score, 100), 0))

        # ======================
        # SIGNAL
        # ======================
        if score >= 80:
            signal = "GOLD"
        elif score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "HOLD"
        else:
            signal = "WATCH"

        # ======================
        # CONFIDENCE
        # ======================
        if payout and payout < 0.75 and cagr and cagr > 5:
            conf = "high"
        elif payout and payout < 1.0:
            conf = "med"
        else:
            conf = "low"

        # ======================
        # WHY
        # ======================
        why = []
        if t.replace(".CO","") in DIVIDEND_KINGS:
            why.append("Dividend King")
        if t.replace(".CO","") in DIVIDEND_ARISTOCRATS:
            why.append("Dividend Aristocrat")
        if yield_pct and yield_pct > 4:
            why.append("High Yield")
        if cagr and cagr > 7:
            why.append("Strong Growth")
        if payout and payout > 1:
            why.append("Payout Risk")

        rows.append({
            "Ticker": t,
            "Name": info.get("longName"),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Currency": info.get("currency"),
            "Price": round(price,2) if price else None,
            "DividendYield_%": yield_pct,
            "DivCAGR_5Y": cagr,
            "Score": score,
            "Signal": signal,
            "Conf": conf,
            "Why": ", ".join(why)
        })

    except Exception as e:
        print(f"Skip {t}: {e}")

# ======================
# SAVE
# ======================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} rows → {OUTPUT_FILE}")
