import yfinance as yf
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
TICKERS = [
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","WMT","COST","HD","LOW",
    "UNH","V","MA","ABBV","AVGO","TXN","QCOM","VZ","T","BCE.TO","RCI-B.TO",
    "NVO","NOVO-B.CO","DNB.OL","ORK.OL","TRV","ALL"
]

OUT_PATH = "data/screener_results.csv"

# ----------------------------
# HELPERS
# ----------------------------
def safe(v):
    return None if v in [None, float("nan")] else v

def calc_years_growing(div_series: pd.Series) -> int:
    if div_series is None or len(div_series) < 2:
        return 0
    years = div_series.resample("Y").sum()
    cnt = 0
    for i in range(len(years)-1, 0, -1):
        if years.iloc[i] > years.iloc[i-1]:
            cnt += 1
        else:
            break
    return cnt

def calc_cagr_5y(div_series: pd.Series) -> float:
    if div_series is None:
        return 0.0
    years = div_series.resample("Y").sum()
    if len(years) < 6:
        return 0.0
    start = years.iloc[-6]
    end = years.iloc[-1]
    if start <= 0:
        return 0.0
    return round(((end / start) ** (1/5) - 1) * 100, 2)

def score_row(yield_pct, payout, roe, years_growing, cagr):
    score = 0
    if yield_pct >= 2: score += 15
    if yield_pct >= 4: score += 10
    if payout and payout < 70: score += 15
    if roe and roe >= 15: score += 20
    if years_growing >= 5: score += 20
    if cagr >= 5: score += 20
    return min(score, 100)

def signal_from_score(score):
    if score >= 80: return "GOLD"
    if score >= 60: return "BUY"
    if score >= 40: return "HOLD"
    return "WATCH"

# ----------------------------
# MAIN
# ----------------------------
rows = []

for t in TICKERS:
    try:
        tk = yf.Ticker(t)
        info = tk.info
        divs = tk.dividends

        price = safe(info.get("currentPrice"))
        dy = safe(info.get("dividendYield"))
        dy = round(dy * 100, 2) if dy else 0.0
        payout = round(info.get("payoutRatio", 0) * 100, 2) if info.get("payoutRatio") else 0.0
        roe = round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else 0.0

        years_growing = calc_years_growing(divs)
        cagr_5y = calc_cagr_5y(divs)

        score = score_row(dy, payout, roe, years_growing, cagr_5y)
        signal = signal_from_score(score)

        rows.append({
            "Ticker": t,
            "Name": info.get("shortName", t),
            "Country": info.get("country"),
            "Sector": info.get("sector"),
            "Price": price,
            "DividendYield_%": dy,
            "PayoutRatio_%": payout,
            "ROE_%": roe,
            "YearsGrowing": years_growing,
            "DivCAGR_5Y_%": cagr_5y,
            "Score": score,
            "Signal": signal
        })

    except Exception as e:
        print(f"Error {t}: {e}")

# FAIL-SAFE: skriv altid CSV
df = pd.DataFrame(rows)
if df.empty:
    df = pd.DataFrame([{
        "Ticker":"ERROR",
        "Name":"No data",
        "Country":None,
        "Sector":None,
        "Price":0,
        "DividendYield_%":0,
        "PayoutRatio_%":0,
        "ROE_%":0,
        "YearsGrowing":0,
        "DivCAGR_5Y_%":0,
        "Score":0,
        "Signal":"WATCH"
    }])

df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows → {OUT_PATH}")
