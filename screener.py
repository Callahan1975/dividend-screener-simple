import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
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

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_float(v, pct=False):
    try:
        val = float(v)
        if pct:
            val *= 100
        return round(val, 2)
    except Exception:
        return None


def normalize_dividend_yield(raw):
    if raw is None:
        return None
    try:
        y = float(raw)
        if y <= 1:
            y *= 100
        if y <= 0 or y > MAX_REASONABLE_YIELD:
            return None
        return round(y, 2)
    except Exception:
        return None


def normalize_payout_ratio(raw):
    if raw is None:
        return None
    try:
        p = float(raw)
        if p <= 0 or p > MAX_REASONABLE_PAYOUT:
            return None
        return round(p, 2)
    except Exception:
        return None


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


# -------------------------------------------------
# MAIN
# -------------------------------------------------
rows = []
generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

for ticker in TICKERS:
    print(f"Processing {ticker}")
    stock = yf.Ticker(ticker)
    flags = []

    # ---------- INFO (må aldrig stoppe tickeren)
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    price = safe_float(info.get("currentPrice"))
    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))

    roe = safe_float(info.get("returnOnEquity"), pct=True)
    roa = safe_float(info.get("returnOnAssets"), pct=True)

    dividend_yield = normalize_dividend_yield(info.get("dividendYield"))
    payout_ratio = normalize_payout_ratio(info.get("payoutRatio"))

    if dividend_yield is None:
        flags.append("YieldMissingOrInvalid")
    if payout_ratio is None:
        flags.append("PayoutMissingOrInvalid")

    # ---------- DIVIDENDS (robust)
    dividends = None
    try:
        raw_div = stock.dividends
        if raw_div is not None and not raw_div.empty:
            dividends = raw_div.resample("Y").sum()
    except Exception:
        dividends = None

    years_growing = len(dividends[dividends > 0]) if dividends is not None else None
    div_cagr_5y = calc_dividend_cagr(dividends, 5)
    dividend_class = classify_dividend(years_growing)

    # ---------- VALUE METRICS
    sector = info.get("sector")
    fair_pe = SECTOR_FAIR_PE.get(sector)
    fair_value = round(eps * fair_pe, 2) if eps and fair_pe else None
    upside = round((fair_value / price - 1) * 100, 2) if price and fair_value else None

    # ---------- PRICE / FCF
    price_to_fcf = None
    try:
        fcf = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if price and fcf and shares:
            price_to_fcf = round(price / (fcf / shares), 2)
    except Exception:
        price_to_fcf = None

    # ---------- FINVIZ STYLE FLAGS
    Value_LowPE = pe is not None and pe < 15
    Value_LowForwardPE = forward_pe is not None and forward_pe < 15
    Value_LowFCF = price_to_fcf is not None and price_to_fcf < 15
    Quality_ROE_10p = roe is not None and roe > 10

    # ---------- SIGNAL ENGINE (v4)
    score = 0

    if Value_LowPE:
        score += 1
    if Value_LowForwardPE:
        score += 1
    if Value_LowFCF:
        score += 1
    if Quality_ROE_10p:
        score += 2
    if years_growing and years_growing >= 10:
        score += 1
    if payout_ratio and payout_ratio < 75:
        score += 1

    signal = "WATCH"
    confidence = "Low"

    if score >= 4:
        signal = "BUY"
        confidence = "Medium"

    if score >= 6 and upside and upside > 10:
        signal = "GOLD"
        confidence = "High"

    if upside is not None and upside < 0:
        signal = "HOLD"
        confidence = "Medium"

    # ---------- ROW
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
        "ForwardPE": forward_pe,
        "PriceToFCF": price_to_fcf,
        "ROE_%": roe,
        "ROA_%": roa,
        "FairPE": fair_pe,
        "FairValue": fair_value,
        "Upside_%": upside,
        "Value_LowPE": Value_LowPE,
        "Value_LowForwardPE": Value_LowForwardPE,
        "Value_LowFCF": Value_LowFCF,
        "Quality_ROE_10p": Quality_ROE_10p,
        "Signal": signal,
        "Confidence": confidence,
        "Flags": ";".join(flags) if flags else ""
    })

# -------------------------------------------------
# EXPORT (aldrig tom)
# -------------------------------------------------
df = pd.DataFrame(rows)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
