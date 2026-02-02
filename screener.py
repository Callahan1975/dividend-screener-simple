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
    tickers |= set(div
