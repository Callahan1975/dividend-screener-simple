from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

TICKERS_FILE = Path("tickers.txt")

OUT_DIR = Path("data/screener_results")
OUT_CSV = OUT_DIR / "screener_results.csv"

# ---- Output columns (locked schema) ----
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
    "YearsPaying",
    "YearsGrowing",
    "DividendClass",
    "DivCAGR_5Y_%",
    "LastDivDate",
    "FairPE",
    "FairValue",
    "Upside_%",
    "Score",
    "Signal",
    "Confidence",
    "Why",
]


# ---------------------------
# Helpers
# ---------------------------

def read_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow inline comments: TICKER # comment
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            tickers.append(line)

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_round(x: Optional[float], ndigits: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except Exception:
        return None


def clamp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None:
        return None
    return max(lo, min(hi, x))


def normalize_percent(x: Optional[float]) -> Optional[float]:
    """
    Normalize to percent points.
    If value <= 1.5 -> treat as fraction (0.034) => 3.4%
    else treat as already percent points.
    """
    if x is None:
        return None
    if x <= 1.5:
        return x * 100.0
    return x


def fetch_info_and_dividends(ticker: str) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
    """
    Returns (info, dividends_series).
    dividends_series is a pandas Series with DatetimeIndex.
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    try:
        divs = t.dividends
        if divs is not None and len(divs) == 0:
            divs = None
    except Exception:
        divs = None

    return info, divs


def pick_price(info: Dict[str, Any]) -> Optional[float]:
    for k in ("currentPrice", "regularMarketPrice", "previousClose"):
        v = to_float(info.get(k))
        if v and v > 0:
            return v
    return None


def pick_pe(info: Dict[str, Any]) -> Optional[float]:
    for k in ("trailingPE", "forwardPE"):
        v = to_float(info.get(k))
        if v and v > 0:
            return v
    return None


def pick_eps(info: Dict[str, Any]) -> Optional[float]:
    for k in ("forwardEps", "trailingEps"):
        v = to_float(info.get(k))
        if v and v > 0:
            return v
    return None


def pick_payout_ratio_percent(info: Dict[str, Any]) -> Optional[float]:
    v = to_float(info.get("payoutRatio"))
    if v is None:
        return None
    v = normalize_percent(v)
    if v is not None and v >= 0:
        return v
    return None


def pick_dividend_yield_percent(info: Dict[str, Any], price: Optional[float], divs: Optional[pd.Series]) -> Optional[float]:
    """
    Robust yield (fixes the 40%-120% nonsense):
    1) Use trailingAnnualDividendRate (cash amount per year) / price
    2) Else compute trailing 12 months (TTM) dividends from dividends series / price
    3) Else fallback to yahoo yield fields (normalized)
    """
    # 1) annual dividend rate / price
    if price and price > 0:
        rate = to_float(info.get("trailingAnnualDividendRate"))
        if rate is not None and rate >= 0:
            return (rate / price) * 100.0

        # 2) TTM from dividends series
        if divs is not None and len(divs) > 0:
            try:
                divs_clean = divs.dropna()
                if len(divs_clean) > 0:
                    if not isinstance(divs_clean.index, pd.DatetimeIndex):
                        divs_clean.index = pd.to_datetime(divs_clean.index)

                    end = divs_clean.index.max()
                    start = end - pd.Timedelta(days=365)
                    ttm = float(divs_clean[divs_clean.index > start].sum())
                    if ttm >= 0:
                        return (ttm / price) * 100.0
            except Exception:
                pass

    # 3) fallback yahoo fields
    for k in ("dividendYield", "trailingAnnualDividendYield"):
        v = to_float(info.get(k))
        if v is None:
            continue
        v = normalize_percent(v)
        if v is not None and v >= 0:
            return v

    return None


def sector_is_reit_like(sector: str, industry: str) -> bool:
    s = (sector or "").lower()
    i = (industry or "").lower()
    return ("real estate" in s) or ("reit" in i) or ("reit" in s)


def dividend_class(years_growing: Optional[int]) -> str:
    """
    Classic labels:
      King:       50+ years growing dividends
      Aristocrat: 25-49
      Contender:  10-24
    """
    if years_growing is None:
        return ""
    if years_growing >= 50:
        return "King"
    if years_growing >= 25:
        return "Aristocrat"
    if years_growing >= 10:
        return "Contender"
    return ""


def compute_fair_pe(pe: Optional[float], sector: str) -> Optional[float]:
    """
    Stable “fair PE” heuristic.
    If PE exists: clamp into a reasonable band as proxy.
    If missing: sector defaults.
    """
    sector_l = (sector or "").lower()
    defaults = {
        "technology": 22.0,
        "consumer defensive": 20.0,
        "consumer cyclical": 18.0,
        "healthcare": 18.0,
        "financial services": 12.0,
        "industrials": 16.0,
        "utilities": 16.0,
        "energy": 12.0,
        "basic materials": 12.0,
        "communication services": 18.0,
        "real estate": 16.0,
    }

    if pe and pe > 0:
        return clamp(pe, 8.0, 28.0)

    for key, val in defaults.items():
        if key in sector_l:
            return val
    return 18.0


def compute_upside(price: Optional[float], eps: Optional[float], fair_pe: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (fair_value, upside_%)
    upside_% clamped to [-50, +100]
    """
    if not price or price <= 0:
        return None, None
    if not eps or eps <= 0:
        return None, None
    if not fair_pe or fair_pe <= 0:
        return None, None

    fair_value = eps * fair_pe
    upside = (fair_value / price - 1.0) * 100.0
    upside = clamp(upside, -50.0, 100.0)
    return fair_value, upside


def compute_dividend_stats(divs: Optional[pd.Series]) -> Tuple[Optional[int], Optional[int], Optional[float], str]:
    """
    Returns:
      YearsPaying: number of years with positive annual dividends
      YearsGrowing: consecutive annual increases (streak length, at least 1 if paying)
      DivCAGR_5Y_%: 5y CAGR of annual dividends (if enough data)
      LastDivDate: last dividend date YYYY-MM-DD
    """
    if divs is None or len(divs) == 0:
        return None, None, None, ""

    try:
        divs = divs.dropna()
        if len(divs) == 0:
            return None, None, None, ""

        if not isinstance(divs.index, pd.DatetimeIndex):
            divs.index = pd.to_datetime(divs.index)

        last_date = divs.index.max().strftime("%Y-%m-%d")

        annual = divs.resample("YE").sum()
        annual.index = annual.index.year
        annual = annual[annual > 0]

        if len(annual) == 0:
            return None, None, None, last_date

        years_paying = int(len(annual))

        years_sorted = annual.sort_index()
        vals = years_sorted.values

        growing_increases = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > vals[i - 1]:
                growing_increases += 1
            else:
                break

        if years_paying >= 2 and growing_increases >= 1:
            years_growing = growing_increases + 1
        else:
            years_growing = 1

        cagr_5y = None
        if len(years_sorted) >= 6:
            start = float(years_sorted.iloc[-6])
            end = float(years_sorted.iloc[-1])
            if start > 0 and end > 0:
                cagr_5y = ((end / start) ** (1 / 5) - 1) * 100.0

        return years_paying, int(years_growing), cagr_5y, last_date
    except Exception:
        return None, None, None, ""


def compute_score_signal_confidence(
    sector: str,
    industry: str,
    yield_pct: Optional[float],
    payout_pct: Optional[float],
    pe: Optional[float],
    upside_pct: Optional[float],
    eps: Optional[float],
    price: Optional[float],
    years_growing: Optional[int],
    div_cagr_5y: Optional[float],
) -> Tuple[int, str, str, str]:
    """
    Returns (Score 0..100, Signal, Confidence, Why)
    """

    flags: List[str] = []
    score = 50
    conf = 100

    # ---- data quality penalties ----
    if price is None:
        conf -= 40
        flags.append("Missing price")
    if eps is None:
        conf -= 20
        flags.append("Missing EPS")
    if pe is None:
        conf -= 10
        flags.append("Missing PE")
    if yield_pct is None:
        conf -= 10
        flags.append("Missing yield")
    if payout_pct is None:
        conf -= 10
        flags.append("Missing payout")

    # ---- dividend streak / growth ----
    div_class = dividend_class(years_growing)

    # small quality boost (NOT too strong)
    if years_growing is None:
        conf -= 5
    else:
        if years_growing >= 50:
            score += 6
            flags.append("Dividend King")
        elif years_growing >= 25:
            score += 4
            flags.append("Dividend Aristocrat")
        elif years_growing >= 10:
            score += 2
            flags.append("Dividend Contender")
        elif years_growing >= 5:
            score += 1
            flags.append("Dividend streak 5+")

    if div_cagr_5y is not None:
        if div_cagr_5y >= 10:
            score += 6
            flags.append("Div growth strong")
        elif div_cagr_5y >= 5:
            score += 3
            flags.append("Div growth ok")

    # ---- yield sanity ----
    if yield_pct is not None:
        if yield_pct > 20:
            conf -= 20
            score -= 10
            flags.append("Yield outlier")
        elif yield_pct > 10:
            conf -= 10
            score += 3
            flags.append("High yield")
        elif 2 <= yield_pct <= 6:
            score += 8
        elif yield_pct < 1:
            score -= 2

    # ---- payout sanity (sector aware) ----
    reit_like = sector_is_reit_like(sector, industry)
    if payout_pct is not None:
        if payout_pct > 140 and not reit_like:
            score -= 18
            conf -= 15
            flags.append("Payout very high")
        elif payout_pct > 100 and not reit_like:
            score -= 10
            conf -= 10
            flags.append("Payout high")
        elif payout_pct > 120 and reit_like:
            score -= 6
            conf -= 6
            flags.append("Payout high (REIT)")
        elif payout_pct < 80:
            score += 8

    # ---- valuation via upside ----
    if upside_pct is None:
        conf -= 10
        flags.append("No upside calc")
    else:
        if upside_pct >= 20:
            score += 18
            flags.append("Undervalued")
        elif upside_pct >= 10:
            score += 12
            flags.append("Good upside")
        elif upside_pct >= 5:
            score += 6
        elif upside_pct <= -15:
            score -= 12
            flags.append("Overvalued")
        elif upside_pct <= -5:
            score -= 6

    # ---- PE sanity ----
    if pe is not None:
        if pe > 40:
            score -= 6
            flags.append("PE very high")
        elif pe < 6:
            score -= 4
            flags.append("PE very low")

    score = int(max(0, min(100, score)))

    # ---- Confidence thresholds (keep as before to avoid breaking your view) ----
    # If you want more spread later, change to 90/70 thresholds.
    if conf >= 80:
        confidence = "High"
    elif conf >= 60:
        confidence = "Med"
    else:
        confidence = "Low"

    # ---- Signal ----
    if confidence == "Low":
        signal = "WATCH"
    else:
        has_div_growth = (years_growing is not None and years_growing >= 5) or (div_cagr_5y is not None and div_cagr_5y >= 5)

        if score >= 88 and (upside_pct is not None and upside_pct >= 12) and ("Payout very high" not in flags) and has_div_growth:
            signal = "GOLD"
        elif score >= 75 and (upside_pct is not None and upside_pct >= 5):
            signal = "BUY"
        elif score >= 60:
            signal = "HOLD"
        else:
            signal = "WATCH"

    # ---- Why: keep short (1 warning + 1 positive) ----
    why_parts: List[str] = []

    for p in ("Yield outlier", "Payout very high", "Payout high", "Overvalued", "Missing EPS", "No upside calc"):
        if p in flags:
            why_parts.append(p)
            break

    # Prefer "King/Aristocrat/Contender" as the positive if present
    for p in ("Dividend King", "Dividend Aristocrat", "Dividend Contender", "Undervalued", "Good upside", "Div growth strong", "Div growth ok"):
        if p in flags and p not in why_parts:
            why_parts.append(p)
            break

    why = " / ".join(why_parts) if why_parts else ""

    return score, signal, confidence, why


def to_row(ticker: str, info: Dict[str, Any], divs: Optional[pd.Series]) -> Dict[str, Any]:
    generated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    name = info.get("shortName") or info.get("longName") or ""
    country = info.get("country") or ""
    currency = info.get("currency") or ""
    exchange = info.get("exchange") or info.get("fullExchangeName") or ""
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""

    price = pick_price(info)
    pe = pick_pe(info)
    eps = pick_eps(info)

    # robust yield calculation
    div_yield_pct = pick_dividend_yield_percent(info, price, divs)
    payout_pct = pick_payout_ratio_percent(info)

    years_paying, years_growing, div_cagr_5y, last_div_date = compute_dividend_stats(divs)
    div_class = dividend_class(years_growing)

    fair_pe = compute_fair_pe(pe, sector)
    fair_value, upside_pct = compute_upside(price, eps, fair_pe)

    score, signal, confidence, why = compute_score_signal_confidence(
        sector=sector,
        industry=industry,
        yield_pct=div_yield_pct,
        payout_pct=payout_pct,
        pe=pe,
        upside_pct=upside_pct,
        eps=eps,
        price=price,
        years_growing=years_growing,
        div_cagr_5y=div_cagr_5y,
    )

    row: Dict[str, Any] = {
        "GeneratedUTC": generated,
        "Ticker": ticker,
        "Name": name,
        "Country": country,
        "Currency": currency,
        "Exchange": exchange,
        "Sector": sector,
        "Industry": industry,

        "Price": safe_round(price, 2),
        "DividendYield_%": safe_round(div_yield_pct, 2),
        "PayoutRatio_%": safe_round(payout_pct, 1),
        "PE": safe_round(pe, 1),

        "YearsPaying": years_paying if years_paying is not None else "",
        "YearsGrowing": years_growing if years_growing is not None else "",
        "DividendClass": div_class,

        "DivCAGR_5Y_%": safe_round(div_cagr_5y, 1),
        "LastDivDate": last_div_date,

        "FairPE": safe_round(fair_pe, 1),
        "FairValue": safe_round(fair_value, 2),
        "Upside_%": safe_round(upside_pct, 1),

        "Score": score,
        "Signal": signal,
        "Confidence": confidence,
        "Why": why,
    }

    # ensure all columns exist
    for c in COLUMNS:
        row.setdefault(c, "")
    return row


def main() -> None:
    tickers = read_tickers(TICKERS_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            info, divs = fetch_info_and_dividends(t)
            rows.append(to_row(t, info, divs))
        except Exception as e:
            rows.append({c: "" for c in COLUMNS} | {
                "GeneratedUTC": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": t,
                "Signal": "WATCH",
                "Confidence": "Low",
                "Score": 0,
                "Why": f"Error: {type(e).__name__}",
            })

        # be nice to Yahoo
        time.sleep(0.4)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            out = {c: r.get(c, "") for c in COLUMNS}
            w.writerow(out)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
def infer_country_from_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".CO"):
        return "Denmark"
    if t.endswith(".ST"):
        return "Sweden"
    if t.endswith(".TO"):
        return "Canada"
    return "United States"
def infer_country_from_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".CO"):
        return "Denmark"
    if t.endswith(".ST"):
        return "Sweden"
    if t.endswith(".TO"):
        return "Canada"
    return "United States"


if __name__ == "__main__":
    main()
