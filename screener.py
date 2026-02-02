import os
import pandas as pd
import yfinance as yf

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

# Your confirmed structure:
DIV_HIST_PATH = "data/dividend_history/dividend_history.csv"

# Streak file: your screenshot suggests it's here:
DIV_STREAK_PRIMARY = "data/dividend_streak.csv"
# but we also support this location if you prefer:
DIV_STREAK_FALLBACK = "data/dividend_history/dividend_streak.csv"

OUTPUT_CSV = "data/screener_results.csv"

# =========================
# LOAD INPUT FILES
# =========================
def load_dividend_history() -> pd.DataFrame:
    if os.path.exists(DIV_HIST_PATH):
        df = pd.read_csv(DIV_HIST_PATH)
        # Expect: Ticker,Year,Dividend
        for c in ["Ticker", "Year", "Dividend"]:
            if c not in df.columns:
                raise ValueError(f"dividend_history.csv missing column: {c}")
        return df
    return pd.DataFrame(columns=["Ticker", "Year", "Dividend"])

def load_dividend_streak() -> pd.DataFrame:
    path = None
    if os.path.exists(DIV_STREAK_PRIMARY):
        path = DIV_STREAK_PRIMARY
    elif os.path.exists(DIV_STREAK_FALLBACK):
        path = DIV_STREAK_FALLBACK

    if path:
        df = pd.read_csv(path)
        # Expect: Ticker,DividendStreak
        for c in ["Ticker", "DividendStreak"]:
            if c not in df.columns:
                raise ValueError(f"{path} missing column: {c}")
        return df
    return pd.DataFrame(columns=["Ticker", "DividendStreak"])

div_hist = load_dividend_history()
div_streak = load_dividend_streak()

# =========================
# METRICS
# =========================
def years_growing_window(ticker: str) -> int:
    """Consecutive YoY increases within the window present in dividend_history.csv (e.g., 2019-2024)."""
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 2:
        return 0
    vals = df["Dividend"].astype(float).values
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1]:
            count += 1
        else:
            break
    return int(count)

def div_cagr_5y(ticker: str) -> float:
    """5-year CAGR based on last 6 annual dividend points in the local CSV."""
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 6:
        return 0.0
    recent = df.tail(6)
    start = float(recent.iloc[0]["Dividend"])
    end = float(recent.iloc[-1]["Dividend"])
    if start <= 0 or end <= 0:
        return 0.0
    cagr = (end / start) ** (1 / 5) - 1
    return round(cagr * 100, 2)

def dividend_streak_value(ticker: str) -> int:
    r = div_streak[div_streak["Ticker"] == ticker]
    if len(r) == 0:
        return 0
    return int(float(r.iloc[0]["DividendStreak"]))

def dividend_class(streak: int) -> str:
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
            "Name": info.get("shortName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": info.get("currentPrice", 0),

            "DividendYield_%": round((info.get("dividendYield", 0) or 0) * 100, 2),
            "PayoutRatio_%": round((info.get("payoutRatio", 0) or 0) * 100, 2),
            "ROE_%": round((info.get("returnOnEquity", 0) or 0) * 100, 2),

            # Window metrics (from your local annual CSV)
            "YearsGrowing": years_growing_window(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),

            # Authoritative streak + class
            "DividendStreak": streak,
            "DividendClass": dividend_class(streak),

            # Keep as-is for now (next phase will improve)
            "Score": float(info.get("recommendationMean", 0) or 0),
            "Signal": info.get("recommendationKey", "WATCH").upper()
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        rows.append({
            "Ticker": ticker,
            "Name": "",
            "Country": "",
            "Sector": "",
            "Price": 0,
            "DividendYield_%": 0,
            "PayoutRatio_%": 0,
            "ROE_%": 0,
            "YearsGrowing": years_growing_window(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),
            "DividendStreak": dividend_streak_value(ticker),
            "DividendClass": dividend_class(dividend_streak_value(ticker)),
            "Score": 0,
            "Signal": "WATCH"
        })

df = pd.DataFrame(rows)

# Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV} ({len(df)} rows)")
