import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# -------------------------
# CONFIG
# -------------------------
TICKERS = [
    "AAPL", "JNJ", "MSFT", "KO", "PG"
]

OUTPUT_PATH = "data/screener_results.csv"

MAX_REASONABLE_YIELD = 15.0
MAX_REASONABLE_PAYOUT = 110.0

SECTOR_FAIR_PE = {
    "Technology": 22,
    "Healthcare": 20,
    "Consumer Defensive": 20,
    "Consumer Cyclical": 20,
    "Financial Services": 12,
    "Energy": 12,
    "Utilities": 16,
    "Industrials": 18,
    "Basic Materials": 14,
    "Real Estate": 16,
    "Communication Services": 18
}

# -------------------------
# HELPERS
# -------------------------
def safe_float(v):
    try:
        return round(float(v), 4)
    except Exception:
        return None


def normalize_dividend_yield(raw):
    try:
        if raw is None:
            return None
        y = float(raw)
    except Exception:
        return None

    # ratio → %
    if y <= 1:
        y *= 100

    if y <= 0 or y > MAX_REASONABLE_YIELD:
        return None

    return round(y, 2)


def normalize_payout_ratio(raw):
    try:
        if raw is None:
            return None
        p = float(raw)
    except Exception:
        return None

    if p <= 0 or p > MAX_REASONABLE_PAYOUT:
        return None

    return round(p, 2)


def calc_dividend_cagr(dividends, years=5):
    if dividends is None or len(dividends) < years + 1:
        return None
    try:
        start = dividends.iloc[-years - 1]
        end = dividends.iloc[-1]
        if start <= 0 or end <= 0:
            return None
        return round(((end / start) ** (1 / years) - 1) * 100, 2)
    except Exception:
        return None


def classify_dividend(years):
    if years is None:
        return None
    if years >= 50:
        return "King"
    if years >= 25:
        return "Aristocrat"
    if years >= 10:
        return "Contender"
    return None


# -------------------------
# MAIN
# -------------------------
rows = []
generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

for ticker in TICKERS:
    print(f"Processing {ticker}")
    stock = yf.Ticker(ticker)
    flags = []

    # ---- INFO (må aldrig stoppe tickeren)
    try:
        info = stock.info
    except Exception as e:
        print(f"Info fetch failed for {ticker}: {e}")
        info = {}

    price = safe_float(info.get("currentPrice"))
    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))

    dividend_yield = normalize_dividend_yield(info.get("dividendYield"))
    payout_ratio = normalize_payout_ratio(info.get("payoutRatio"))

    if dividend_yield is None:
        flags.append("YieldMissingOrInvalid")
    if payout_ratio is None:
        flags.append("PayoutMissingOrInvalid")

    # ---- DIVIDENDS (må ALDRIG være fatal)
    dividends = None
    try:
        raw_div = stock.dividends
        if raw_div is not None and not raw_div.empty:
            dividends = raw_div.resample("Y").sum()
    except Exception as e:
        print(f"Dividend history issue for {ticker}: {e}")
        dividends = None

    years_growing = len(dividends[dividends > 0]) if dividends is not None else None
    div_cagr_5y = calc_dividend_cagr(dividends, 5)
    dividend_class = classify_dividend(years_growing)

    # ---- FAIR VALUE
    sector = info.get("sector")
    fair_pe = SECTOR_FAIR_PE.get(sector)
    fair_value = round(eps * fair_pe, 2) if eps and fair_pe else None
    upside = round((fair_value / price - 1) * 100, 2) if price and fair_value else None

    # ---- SIGNAL LOGIC
    signal = "WATCH"
    confidence = "Low"

    if upside is not None and upside < 0:
        signal = "HOLD"
        confidence = "Medium"

    if upside is not None and upside > 20 and payout_ratio and payout_ratio < 80:
        signal = "BUY"
        confidence = "Medium"

    if (
        upside is not None and upside > 30 and
        dividend_yield is not None and dividend_yield >= 2 and
        years_growing is not None and years_growing >= 10
    ):
        signal = "GOLD"
        confidence = "High"

    # ---- ROW
    rows.append({
        "GeneratedUTC": generated_utc,
        "Ticker": ticker,
        "Name": info.get("shortName"),
        "Country": info.get("country"),
        "Sector": sector,
        "Industry": info.get("industry"),
        "Price": price,
        "DividendYield_%": dividend_yield,
        "PayoutRatio_%": payout_ratio,
        "DivCAGR_5Y_%": div_cagr_5y,
        "YearsGrowing": years_growing,
        "DividendClass": dividend_class,
        "PE": pe,
        "FairPE": fair_pe,
        "FairValue": fair_value,
        "Upside_%": upside,
        "Signal": signal,
        "Confidence": confidence,
        "Flags": ";".join(flags) if flags else ""
    })

# -------------------------
# EXPORT (CSV MÅ ALDRIG VÆRE TOM)
# -------------------------
df = pd.DataFrame(rows)

ordered_columns = [
    "GeneratedUTC",
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Industry",
    "Price",
    "DividendYield_%",
    "PayoutRatio_%",
    "DivCAGR_5Y_%",
    "YearsGrowing",
    "DividendClass",
    "PE",
    "FairPE",
    "FairValue",
    "Upside_%",
    "Signal",
    "Confidence",
    "Flags"
]

df = df.reindex(columns=ordered_columns)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
