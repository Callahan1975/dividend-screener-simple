import pandas as pd
import yfinance as yf
import os

# =========================
# CONFIG
# =========================
TICKERS = [
    "AAPL","ABBV","ADP","AMGN","ARCC",
    "ASSA-B.ST","ATCO-A.ST","ATCO-B.ST",
    "BMO.TO","BNS.TO","CARL-B.CO","CAT","CL",
    "CNQ.TO","COST","CVX","D","DNB.OL","DUK",
    "ED","EMR","ENB.TO","EQNR.OL","JNJ","KO",
    "MSFT","NOVO-B.CO","O","ORSTED.CO","PEP",
    "PG","RY.TO","SEB-A.ST","SHB-A.ST","SWED-A.ST",
    "TD.TO","TEL2-B.ST","TELIA.ST","TRP.TO",
    "VOLV-B.ST","WM","XOM"
]

DIV_HIST_PATH = "data/dividend_history/dividend_history.csv"
DIV_STREAK_PATH = "data/dividend_history/dividend_streak.csv"

OUTPUT_PATHS = [
    "data/screener_results/screener_results.csv",
    "docs/data/screener_results/screener_results.csv"
]

# =========================
# LOAD DATA
# =========================
div_hist = pd.read_csv(DIV_HIST_PATH) if os.path.exists(DIV_HIST_PATH) else pd.DataFrame(columns=["Ticker","Year","Dividend"])
div_streak = pd.read_csv(DIV_STREAK_PATH) if os.path.exists(DIV_STREAK_PATH) else pd.DataFrame(columns=["Ticker","DividendStreak"])

# =========================
# METRICS
# =========================
def years_growing_window(ticker):
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 2:
        return 0
    vals = df["Dividend"].values
    c = 0
    for i in range(len(vals)-1, 0, -1):
        if vals[i] > vals[i-1]:
            c += 1
        else:
            break
    return c

def div_cagr_5y(ticker):
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 6:
        return 0.0
    recent = df.tail(6)
    s, e = recent.iloc[0]["Dividend"], recent.iloc[-1]["Dividend"]
    if s <= 0 or e <= 0:
        return 0.0
    return round(((e / s) ** (1/5) - 1) * 100, 2)

def dividend_streak_value(ticker):
    r = div_streak[div_streak["Ticker"] == ticker]
    return int(r.iloc[0]["DividendStreak"]) if len(r) else 0

def dividend_class(streak):
    if streak >= 50:
        return "King"
    if streak >= 25:
        return "Aristocrat"
    if streak >= 10:
        return "Contender"
    return ""

# =========================
# MAIN
# =========================
rows = []

for ticker in TICKERS:
    try:
        info = yf.Ticker(ticker).info
        streak = dividend_streak_value(ticker)

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName",""),
            "Country": info.get("country",""),
            "Sector": info.get("sector",""),
            "Price": info.get("currentPrice", 0),
            "DividendYield_%": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio", 0) or 0) * 100, 2),
            "ROE_%": round((info.get("returnOnEquity", 0) or 0) * 100, 2),

            # Window-based (from CSV)
            "YearsGrowing": years_growing_window(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),

            # Authoritative
            "DividendStreak": streak,
            "DividendClass": dividend_class(streak),

            "Score": 0,
            "Signal": "WATCH"
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

df = pd.DataFrame(rows)

# =========================
# SAVE
# =========================
for p in OUTPUT_PATHS:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)

print("Fase 2B complete – DividendStreak & DividendClass added.")
