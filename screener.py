from __future__ import annotations

import csv
import datetime as dt
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

# ======================================================
# PATHS
# ======================================================
ROOT = Path(".")
TICKERS_FILE = ROOT / "tickers.txt"
ALIAS_FILE = ROOT / "data" / "ticker_alias.csv"

OUT_DIR = ROOT / "data" / "screener_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "screener_results.csv"

DOCS_DIR = ROOT / "docs" / "data" / "screener_results"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_CSV = DOCS_DIR / "screener_results.csv"

# ======================================================
# LOAD TICKER ALIAS (AUTHORITATIVE SOURCE)
# ======================================================
if not ALIAS_FILE.exists():
    raise FileNotFoundError("data/ticker_alias.csv not found")

alias_df = pd.read_csv(
    ALIAS_FILE,
    comment="#",
    skip_blank_lines=True
).dropna(how="all")

required_cols = {"Ticker", "PrimaryTicker", "Country", "Exchange"}
missing = required_cols - set(alias_df.columns)
if missing:
    raise ValueError(f"Missing columns in ticker_alias.csv: {missing}")

alias_df["Ticker"] = alias_df["Ticker"].str.upper().str.strip()
alias_df["PrimaryTicker"] = alias_df["PrimaryTicker"].str.upper().str.strip()

if alias_df["Ticker"].duplicated().any():
    raise ValueError("Duplicate Ticker values in ticker_alias.csv")

ALIAS: Dict[str, Dict[str, str]] = alias_df.set_index("Ticker").to_dict("index")
print(f"✅ Loaded {len(ALIAS)} ticker aliases")

# ======================================================
# CSV OUTPUT COLUMNS
# ======================================================
COLUMNS = [
    "GeneratedUTC","Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Price","DividendYield_%","PayoutRatio_%",
    "PE","EPS","FairPE","FairValue","Upside_%","DivCAGR_5Y_%",
    "YearsGrowing","DividendClass","Score","Signal","Confidence",
    "Why","Flags"
]

# ======================================================
# HELPERS
# ======================================================
def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize_pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return x * 100 if x <= 1.5 else x

def annual_divs(divs: pd.Series) -> pd.Series:
    if divs is None or divs.empty:
        return pd.Series(dtype=float)
    divs.index = pd.to_datetime(divs.index)
    return divs.resample("Y").sum()

def div_class(years: Optional[int]) -> str:
    if not years:
        return ""
    if years >= 50: return "King"
    if years >= 25: return "Aristocrat"
    if years >= 10: return "Contender"
    return ""

# ======================================================
# CORE ROW BUILDER
# ======================================================
def build_row(ticker: str, ts: str) -> Dict[str, Any]:
    if ticker not in ALIAS:
        raise ValueError(f"{ticker} missing in ticker_alias.csv")

    meta = ALIAS[ticker]
    yahoo = meta["PrimaryTicker"]

    tk = yf.Ticker(yahoo)
    info = tk.info or {}

    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))
    payout = normalize_pct(safe_float(info.get("payoutRatio")))

    try:
        dividends = tk.dividends
    except Exception:
        dividends = None

    yield_pct = normalize_pct(safe_float(info.get("dividendYield")))
    if (yield_pct is None or yield_pct > 40) and dividends is not None and price:
        ltm = dividends.last("365D").sum()
        if ltm > 0:
            yield_pct = (ltm / price) * 100

    years = None
    cagr = None
    if dividends is not None:
        ann = annual_divs(dividends)
        ann.index = ann.index.year
        ann = ann.iloc[:-1] if len(ann) > 2 else ann
        if len(ann) >= 3:
            years = sum(ann.iloc[i] > ann.iloc[i-1] for i in range(len(ann)-1,0,-1))
        if len(ann) >= 6 and ann.iloc[-6] > 0:
            cagr = ((ann.iloc[-1]/ann.iloc[-6])**(1/5)-1)*100

    fair_pe = clamp(pe,10,28) if pe and pe < 80 else 18
    fair_val = eps * fair_pe if eps else None
    upside = ((fair_val/price)-1)*100 if price and fair_val else None

    score = 50
    flags = []

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
    if cagr and cagr > 7:
        score += 8

    score = int(clamp(score,0,100))
    confidence = "High" if score >= 80 and not flags else "Medium"
    signal = "GOLD" if score >= 90 else "BUY" if score >= 80 else "HOLD" if score >= 60 else "WATCH"

    return {
        "GeneratedUTC": ts,
        "Ticker": ticker,
        "Name": info.get("shortName") or info.get("longName") or "",
        "Country": meta["Country"],
        "Currency": info.get("currency",""),
        "Exchange": meta["Exchange"],
        "Sector": info.get("sector",""),
        "Industry": info.get("industry",""),
        "Price": price,
        "DividendYield_%": yield_pct,
        "PayoutRatio_%": payout,
        "PE": pe,
        "EPS": eps,
        "FairPE": fair_pe,
        "FairValue": fair_val,
        "Upside_%": upside,
        "DivCAGR_5Y_%": cagr,
        "YearsGrowing": years,
        "DividendClass": div_class(years),
        "Score": score,
        "Signal": signal,
        "Confidence": confidence,
        "Why": " / ".join(flags) if flags else "—",
        "Flags": " | ".join(flags),
    }

# ======================================================
# MAIN
# ======================================================
def main() -> None:
    tickers = [t.strip().upper() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            rows.append(build_row(t, ts))
        except Exception as e:
            print(f"❌ {t}: {e}")
        time.sleep(0.35)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c,"") for c in COLUMNS})

    DOCS_CSV.write_text(OUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"✅ Generated {len(rows)} rows")

if __name__ == "__main__":
    main()
