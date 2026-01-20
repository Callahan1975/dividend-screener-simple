from __future__ import annotations

import csv
import datetime as dt
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# ----------------------------
# Paths
# ----------------------------
TICKERS_FILE = Path("data/tickers.txt")

OUT_DIR = Path("data/screener_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "screener_results.csv"

DOCS_DATA_DIR = Path("docs/data/screener_results")
DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_CSV = DOCS_DATA_DIR / "screener_results.csv"

# ----------------------------
# ALIAS (SINGLE SOURCE OF TRUTH)
# ----------------------------
ALIAS_FILE = Path("data/ticker_alias.csv")
if not ALIAS_FILE.exists():
    raise FileNotFoundError("ticker_alias.csv is REQUIRED")

alias_df = pd.read_csv(ALIAS_FILE)
alias_df["Ticker"] = alias_df["Ticker"].str.strip()
ALIAS: Dict[str, Dict[str, str]] = alias_df.set_index("Ticker").to_dict("index")

print(f"Loaded {len(ALIAS)} ticker aliases")

# ----------------------------
# Columns
# ----------------------------
COLUMNS = [
    "GeneratedUTC",
    "Ticker",
    "Name",
    "Country",
    "Currency",
    "Exchange",
    "Sector",
    "Industry",
    "Price",
    "DividendYield_%",
    "PayoutRatio_%",
    "PE",
    "EPS",
    "FairPE",
    "FairValue",
    "Upside_%",
    "DivCAGR_5Y_%",
    "YearsGrowing",
    "DividendClass",
    "Score",
    "Signal",
    "Confidence",
    "Why",
    "Flags",
]

# ----------------------------
# Helpers (uændret)
# ----------------------------
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x <= 1.5:
        return x * 100
    return x


def annual_dividends(divs: pd.Series) -> pd.Series:
    if divs is None or divs.empty:
        return pd.Series(dtype=float)
    divs.index = pd.to_datetime(divs.index)
    return divs.resample("Y").sum()


def dividend_class(years: Optional[int]) -> str:
    if years is None:
        return ""
    if years >= 50:
        return "King"
    if years >= 25:
        return "Aristocrat"
    if years >= 10:
        return "Contender"
    return ""


# ----------------------------
# META RESOLUTION (NYT)
# ----------------------------
def resolve_meta(ticker: str, info: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Resolve Country, Exchange, Currency
    Priority:
    1) ticker_alias.csv
    2) yfinance info
    """
    # ALIAS: primary source
    if ticker in ALIAS:
        a = ALIAS[ticker]
        return (
            a.get("Country", ""),
            a.get("Exchange", ""),
            a.get("Currency", ""),
        )

    # fallback (burde ikke ske)
    return (
        info.get("country", "") or "",
        info.get("exchange", "") or "",
        info.get("currency", "") or "",
    )


# ----------------------------
# Core build_row (samme logik)
# ----------------------------
def build_row(ticker: str, generated_utc: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    name = info.get("shortName") or info.get("longName") or ""
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""

    # ALIAS: resolve meta centrally
    country, exchange, currency = resolve_meta(ticker, info)

    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    try:
        dividends = tk.dividends
    except Exception:
        dividends = None

    yield_pct = normalize_percent(safe_float(info.get("dividendYield")))
    if (yield_pct is None or yield_pct > 40) and dividends is not None and price:
        ltm = dividends.last("365D").sum()
        if ltm > 0:
            yield_pct = (ltm / price) * 100

    payout = normalize_percent(safe_float(info.get("payoutRatio")))
    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))

    years_growing = None
    div_cagr = None

    if dividends is not None:
        ann = annual_dividends(dividends)
        ann.index = ann.index.year
        ann = ann.iloc[:-1] if len(ann) > 2 else ann

        if len(ann) >= 3:
            vals = ann.values
            cnt = 0
            for i in range(len(vals) - 1, 0, -1):
                if vals[i] > vals[i - 1]:
                    cnt += 1
                else:
                    break
            years_growing = cnt

        if len(ann) >= 6:
            start, end = ann.iloc[-6], ann.iloc[-1]
            if start > 0:
                div_cagr = ((end / start) ** (1 / 5) - 1) * 100

    fair_pe = clamp(pe, 10, 28) if pe and pe < 80 else 18
    fair_value = eps * fair_pe if eps else None
    upside = ((fair_value / price) - 1) * 100 if price and fair_value else None

    flags = []
    score = 50

    if yield_pct:
        if yield_pct > 30:
            flags.append("Yield outlier")
            score -= 10
        elif 1 <= yield_pct <= 6:
            score += 10

    if payout:
        if payout > 110:
            flags.append("Payout high")
            score -= 10
        elif payout <= 70:
            score += 8

    if upside and upside > 10:
        score += 10

    if div_cagr and div_cagr > 7:
        score += 8

    score = int(clamp(score, 0, 100))

    confidence = "High" if score >= 80 and not flags else "Medium"
    signal = "GOLD" if score >= 90 else "BUY" if score >= 80 else "HOLD" if score >= 60 else "WATCH"

    return {
        "GeneratedUTC": generated_utc,
        "Ticker": ticker,
        "Name": name,
        "Country": country,
        "Currency": currency,
        "Exchange": exchange,
        "Sector": sector,
        "Industry": industry,
        "Price": price,
        "DividendYield_%": yield_pct,
        "PayoutRatio_%": payout,
        "PE": pe,
        "EPS": eps,
        "FairPE": fair_pe,
        "FairValue": fair_value,
        "Upside_%": upside,
        "DivCAGR_5Y_%": div_cagr,
        "YearsGrowing": years_growing,
        "DividendClass": dividend_class(years_growing),
        "Score": score,
        "Signal": signal,
        "Confidence": confidence,
        "Why": " / ".join(flags) if flags else "—",
        "Flags": " | ".join(flags),
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows: List[Dict[str, Any]] = []

    for t in tickers:
        try:
            rows.append(build_row(t, ts))
        except Exception as e:
            print(f"ERROR {t}: {e}")
        time.sleep(0.35)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    DOCS_CSV.write_text(OUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Generated {len(rows)} rows")


if __name__ == "__main__":
    main()
