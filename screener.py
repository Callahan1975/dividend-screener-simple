import os
import pandas as pd
import yfinance as yf

# =========================
# PATHS (match your repo)
# =========================
DIV_HIST_PATH = "data/dividend_history/dividend_history.csv"
DIV_STREAK_PATH = "data/dividend_streak.csv"
OUTPUT_CSV = "data/screener_results.csv"

# =========================
# HELPERS
# =========================
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def normalize_percent(x) -> float:
    """
    Yahoo sometimes returns ratios as:
      - 0.0279 for 2.79%
      - 2.79 for 2.79%
    We normalize to a clean percent number (2.79).
    """
    v = _safe_float(x, 0.0)
    if v <= 0:
        return 0.0
    # If it's a fraction (<=1), convert to %
    if v <= 1.0:
        v *= 100.0
    # Hard sanity clamp: yields above 80% are likely wrong data
    # (you can adjust/remove later)
    if v > 80:
        return round(v / 100.0, 2) if v <= 8000 else 0.0
    return round(v, 2)

def normalize_price(x) -> float:
    v = _safe_float(x, 0.0)
    return round(v, 2) if v > 0 else 0.0

def clean_ticker(t: str) -> str:
    return (t or "").strip().upper()

# =========================
# LOAD INPUT CSVs
# =========================
def load_dividend_history() -> pd.DataFrame:
    if not os.path.exists(DIV_HIST_PATH):
        return pd.DataFrame(columns=["Ticker", "Year", "Dividend"])

    df = pd.read_csv(DIV_HIST_PATH)
    needed = ["Ticker", "Year", "Dividend"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{DIV_HIST_PATH} missing column: {c}")

    df["Ticker"] = df["Ticker"].astype(str).map(clean_ticker)
    df["Year"] = df["Year"].astype(int)
    df["Dividend"] = df["Dividend"].astype(float)
    return df

def load_dividend_streak() -> pd.DataFrame:
    if not os.path.exists(DIV_STREAK_PATH):
        return pd.DataFrame(columns=["Ticker", "DividendStreak"])

    df = pd.read_csv(DIV_STREAK_PATH)
    needed = ["Ticker", "DividendStreak"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{DIV_STREAK_PATH} missing column: {c}")

    df["Ticker"] = df["Ticker"].astype(str).map(clean_ticker)
    df["DividendStreak"] = df["DividendStreak"].astype(int)
    return df

div_hist = load_dividend_history()
div_streak = load_dividend_streak()

# =========================
# BUILD TICKER UNIVERSE (CSV-only)
# =========================
tickers = set()
if len(div_hist):
    tickers |= set(div_hist["Ticker"].dropna().unique())
if len(div_streak):
    tickers |= set(div_streak["Ticker"].dropna().unique())

tickers = sorted([t for t in tickers if t])

if not tickers:
    raise ValueError(
        "No tickers found. Add tickers to data/dividend_history/dividend_history.csv "
        "and/or data/dividend_streak.csv."
    )

# =========================
# METRICS (from local dividend_history.csv)
# =========================
def years_growing_window(ticker: str) -> int:
    df = div_hist[div_hist["Ticker"] == ticker].sort_values("Year")
    if len(df) < 2:
        return 0
    vals = df["Dividend"].values
    count = 0
    # Count consecutive increases from the end
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1]:
            count += 1
        else:
            break
    return int(count)

def div_cagr_5y(ticker: str) -> float:
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
    return int(r.iloc[0]["DividendStreak"]) if len(r) else 0

def dividend_class(streak: int) -> str:
    if streak >= 50:
        return "King"
    if streak >= 25:
        return "Aristocrat"
    if streak >= 10:
        return "Contender"
    return ""

# =========================
# MAIN FETCH
# =========================
rows = []

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        rec_key = (info.get("recommendationKey") or "watch").upper()

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", "") or info.get("longName", ""),
            "Country": info.get("country", ""),
            "Sector": info.get("sector", ""),
            "Price": normalize_price(info.get("currentPrice") or info.get("regularMarketPrice")),

            # ✅ fixed yield normalization (0.0279 -> 2.79, 2.79 -> 2.79)
            "DividendYield_%": normalize_percent(info.get("dividendYield")),

            # Normalize these too (Yahoo often provides fractions)
            "PayoutRatio_%": normalize_percent(info.get("payoutRatio")),
            "ROE_%": normalize_percent(info.get("returnOnEquity")),

            # Local window metrics
            "YearsGrowing": years_growing_window(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),

            # Authoritative streak + class (from your streak CSV)
            "DividendStreak": dividend_streak_value(ticker),
            "DividendClass": dividend_class(dividend_streak_value(ticker)),

            # Leave score/signal simple for now (phase later can replace)
            "Score": round(_safe_float(info.get("recommendationMean"), 0.0), 2),
            "Signal": rec_key if rec_key else "WATCH"
        })

    except Exception as e:
        # Keep row even if Yahoo fails
        print(f"Error processing {ticker}: {e}")
        rows.append({
            "Ticker": ticker,
            "Name": "",
            "Country": "",
            "Sector": "",
            "Price": 0.0,
            "DividendYield_%": 0.0,
            "PayoutRatio_%": 0.0,
            "ROE_%": 0.0,
            "YearsGrowing": years_growing_window(ticker),
            "DivCAGR_5Y_%": div_cagr_5y(ticker),
            "DividendStreak": dividend_streak_value(ticker),
            "DividendClass": dividend_class(dividend_streak_value(ticker)),
            "Score": 0.0,
            "Signal": "WATCH"
        })

df = pd.DataFrame(rows)

# Stable column order (so DataTables never “mixer”)
cols = [
    "Ticker","Name","Country","Sector","Price",
    "DividendYield_%","PayoutRatio_%","ROE_%",
    "YearsGrowing","DivCAGR_5Y_%",
    "DividendStreak","DividendClass",
    "Score","Signal"
]
for c in cols:
    if c not in df.columns:
        df[c] = ""

df = df[cols]

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV} ({len(df)} rows) from CSV-universe ({len(tickers)} tickers).")
