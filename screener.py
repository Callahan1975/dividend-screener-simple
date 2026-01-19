from __future__ import annotations

import csv
import datetime as dt
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# ----------------------------
# Paths
# ----------------------------
TICKERS_FILE = Path("tickers.txt")

OUT_DIR = Path("data/screener_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "screener_results.csv"

DOCS_DATA_DIR = Path("docs/data/screener_results")
DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

DOCS_CSV = DOCS_DATA_DIR / "screener_results.csv"


# ----------------------------
# Columns for CSV / UI
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
    "DividendYield_%",      # percent
    "PayoutRatio_%",        # percent
    "PE",
    "EPS",
    "FairPE",
    "FairValue",
    "Upside_%",             # percent
    "DivCAGR_5Y_%",          # percent
    "YearsGrowing",          # integer
    "DividendClass",         # King/Aristocrat/Contender/""
    "Score",                 # 0-100
    "Signal",                # GOLD/BUY/HOLD/WATCH/ERROR
    "Confidence",            # High/Medium/Low
    "Why",                   # short reason
    "Flags",                 # pipe separated flags
]


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def infer_country_from_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".CO"):
        return "Denmark"
    if t.endswith(".ST"):
        return "Sweden"
    if t.endswith(".TO"):
        return "Canada"
    if t.endswith(".HE"):
        return "Finland"
    return "United States"


def infer_currency_from_ticker(ticker: str) -> Optional[str]:
    t = ticker.upper()
    if t.endswith(".CO"):
        return "DKK"
    if t.endswith(".ST"):
        return "SEK"
    if t.endswith(".TO"):
        return "CAD"
    if t.endswith(".HE"):
        return "EUR"
    return "USD"


def dividend_class(years_growing: Optional[int]) -> str:
    if years_growing is None:
        return ""
    if years_growing >= 50:
        return "King"
    if years_growing >= 25:
        return "Aristocrat"
    if years_growing >= 10:
        return "Contender"
    return ""


def normalize_percent(x: Optional[float]) -> Optional[float]:
    """
    Ensure percent is in 0..100 range if possible.
    Accepts either 0.034 -> 3.4, or 3.4 -> 3.4.
    """
    if x is None:
        return None
    if x < 0:
        return x
    if x <= 1.5:
        return x * 100.0
    return x


def compute_ltm_dividend_yield(dividends: pd.Series, price: Optional[float]) -> Optional[float]:
    """
    LTM dividend yield in percent from dividend history.
    """
    if price is None or price <= 0:
        return None
    if dividends is None or len(dividends) == 0:
        return None
    end = dividends.index.max()
    start = end - pd.Timedelta(days=365)
    ltm = dividends.loc[dividends.index > start].sum()
    if ltm <= 0:
        return None
    return (ltm / price) * 100.0


def annual_dividends(dividends: pd.Series) -> pd.Series:
    if dividends is None or len(dividends) == 0:
        return pd.Series(dtype=float)
    s = dividends.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("Y").sum()  # year-end bins


def compute_years_growing(dividends: pd.Series) -> Optional[int]:
    """
    Count consecutive years of dividend growth using full-year totals.
    Conservative: uses year totals and checks strictly increasing year over year.
    """
    ann = annual_dividends(dividends)
    if ann.empty or len(ann) < 3:
        return None

    # Convert to calendar years and drop current year if incomplete
    ann.index = ann.index.year
    # drop latest year if too small timeframe (conservative)
    # Keep last completed year: we assume current calendar year may be incomplete
    current_year = dt.datetime.utcnow().year
    if current_year in ann.index:
        ann = ann.drop(index=current_year, errors="ignore")

    if len(ann) < 3:
        return None

    ann = ann.sort_index()
    # Walk backwards counting strictly increasing totals
    years = list(ann.index)
    vals = [float(ann.loc[y]) for y in years]

    # Start from last year
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > 0 and vals[i] > vals[i - 1]:
            count += 1
        else:
            break

    return count


def compute_div_cagr(dividends: pd.Series, years: int = 5) -> Optional[float]:
    """
    CAGR of annual dividend totals over 'years' years, percent.
    Uses completed years only. Requires start and end totals > 0.
    """
    ann = annual_dividends(dividends)
    if ann.empty:
        return None
    ann.index = ann.index.year
    current_year = dt.datetime.utcnow().year
    ann = ann.drop(index=current_year, errors="ignore")
    ann = ann.sort_index()
    if len(ann) < years + 1:
        return None

    end_year = ann.index.max()
    start_year = end_year - years
    if start_year not in ann.index or end_year not in ann.index:
        return None

    start_val = float(ann.loc[start_year])
    end_val = float(ann.loc[end_year])
    if start_val <= 0 or end_val <= 0:
        return None

    cagr = (end_val / start_val) ** (1.0 / years) - 1.0
    return cagr * 100.0


def sanity_yield(y: Optional[float]) -> Tuple[Optional[float], List[str]]:
    flags = []
    if y is None:
        return None, flags
    # absurd yields are usually data glitches (or special situations)
    if y > 30:
        flags.append("Yield outlier")
        # keep it but flagged
    if y > 100:
        flags.append("Yield invalid")
    return y, flags


def sanity_payout(p: Optional[float], sector: str) -> Tuple[Optional[float], List[str]]:
    flags = []
    if p is None:
        return None, flags
    if p < 0:
        flags.append("Payout negative")
    # sector-smart thresholds (simple)
    high_thr = 90.0
    if sector in ("Real Estate",):  # REITs often higher
        high_thr = 110.0
    if sector in ("Financial Services",):
        high_thr = 110.0
    if p > high_thr:
        flags.append("Payout high")
    if p > 200:
        flags.append("Payout extreme")
    return p, flags


def score_signal_conf(
    yield_pct: Optional[float],
    payout_pct: Optional[float],
    upside_pct: Optional[float],
    div_cagr_5y: Optional[float],
    years_growing: Optional[int],
    div_class: str,
    flags: List[str],
    has_price: bool,
    has_name: bool,
) -> Tuple[int, str, str, str]:
    """
    Return (score, signal, confidence, why)
    """
    if not has_price or not has_name:
        return 0, "ERROR", "Low", "Missing price/name"

    score = 50

    # Value component (upside)
    if upside_pct is not None:
        if upside_pct >= 50:
            score += 20
        elif upside_pct >= 20:
            score += 15
        elif upside_pct >= 10:
            score += 10
        elif upside_pct >= 0:
            score += 4
        else:
            score -= 8

    # Yield component (prefer 1-6 for quality names, allow higher but flag)
    if yield_pct is not None:
        if 1.0 <= yield_pct <= 6.0:
            score += 10
        elif 0.5 <= yield_pct < 1.0:
            score += 4
        elif 6.0 < yield_pct <= 10.0:
            score += 6
        elif yield_pct > 10.0:
            score -= 8  # likely special situation
    else:
        score -= 4

    # Payout component
    if payout_pct is not None:
        if 0 <= payout_pct <= 70:
            score += 10
        elif 70 < payout_pct <= 90:
            score += 4
        elif payout_pct > 90:
            score -= 8
    else:
        score -= 2

    # Dividend growth component
    if div_cagr_5y is not None:
        if div_cagr_5y >= 12:
            score += 10
        elif div_cagr_5y >= 7:
            score += 7
        elif div_cagr_5y >= 3:
            score += 4
        elif div_cagr_5y < 0:
            score -= 6

    if years_growing is not None:
        if years_growing >= 25:
            score += 6
        elif years_growing >= 10:
            score += 4
        elif years_growing >= 5:
            score += 2

    # Dividend class tiny boost
    if div_class == "King":
        score += 4
    elif div_class == "Aristocrat":
        score += 3
    elif div_class == "Contender":
        score += 2

    # Flag penalties
    if "Yield invalid" in flags:
        score -= 25
    if "Yield outlier" in flags:
        score -= 12
    if "Payout extreme" in flags:
        score -= 20
    if "Payout high" in flags:
        score -= 8

    score = int(clamp(score, 0, 100))

    # Confidence
    # High requires: no big flags + at least price + yield + upside available
    big_flags = any(f in flags for f in ["Yield invalid", "Payout extreme"])
    mid_flags = any(f in flags for f in ["Yield outlier", "Payout high"])

    if big_flags:
        conf = "Low"
    else:
        if yield_pct is not None and upside_pct is not None and not mid_flags:
            conf = "High"
        elif yield_pct is not None or upside_pct is not None:
            conf = "Medium"
        else:
            conf = "Low"

    # Signal
    # GOLD = best candidates (score + upside) and not low confidence
    if conf != "Low" and score >= 90 and (upside_pct or 0) >= 10:
        signal = "GOLD"
    elif conf != "Low" and score >= 80 and (upside_pct or 0) >= 5:
        signal = "BUY"
    elif score >= 60:
        signal = "HOLD"
    else:
        signal = "WATCH"

    # Why (short)
    why_parts = []
    if div_class:
        why_parts.append(f"Dividend {div_class}")
    if upside_pct is not None:
        if upside_pct >= 10:
            why_parts.append("Undervalued")
        elif upside_pct < 0:
            why_parts.append("Overvalued")
        else:
            why_parts.append("Fair value")
    if div_cagr_5y is not None and div_cagr_5y >= 7:
        why_parts.append("Div growth strong")
    elif div_cagr_5y is not None and div_cagr_5y >= 3:
        why_parts.append("Div growth ok")
    if "Payout high" in flags:
        why_parts.append("Payout high")
    if "Yield outlier" in flags:
        why_parts.append("Yield outlier")

    why = " / ".join(why_parts) if why_parts else "—"

    return score, signal, conf, why


# ----------------------------
# Core fetch logic
# ----------------------------
def read_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow inline comment: TICKER # comment
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if s:
            tickers.append(s)

    # de-dupe preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def get_info_safe(t: yf.Ticker) -> Dict[str, Any]:
    # yfinance sometimes errors on .info; keep it safe
    try:
        inf = t.info or {}
        if isinstance(inf, dict):
            return inf
        return {}
    except Exception:
        return {}


def get_fast_price_safe(t: yf.Ticker) -> Optional[float]:
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            # yfinance fast_info keys vary by version
            for k in ["last_price", "lastPrice", "regularMarketPrice"]:
                v = safe_float(fi.get(k)) if hasattr(fi, "get") else None
                if v is not None and v > 0:
                    return v
            v = safe_float(fi.get("previous_close")) if hasattr(fi, "get") else None
            if v is not None and v > 0:
                return v
    except Exception:
        pass
    return None


def get_history_price_fallback(t: yf.Ticker) -> Optional[float]:
    try:
        hist = t.history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        v = safe_float(hist["Close"].dropna().iloc[-1])
        if v is not None and v > 0:
            return v
    except Exception:
        return None
    return None


def build_row(ticker: str, generated_utc: str) -> Dict[str, Any]:
    country = infer_country_from_ticker(ticker)
    currency_guess = infer_currency_from_ticker(ticker)

    try:
        tk = yf.Ticker(ticker)

        info = get_info_safe(tk)
        name = (info.get("shortName") or info.get("longName") or "").strip()
        exchange = (info.get("exchange") or info.get("fullExchangeName") or "").strip()
        sector = (info.get("sector") or "").strip()
        industry = (info.get("industry") or "").strip()
        currency = (info.get("currency") or currency_guess or "").strip()

        price = get_fast_price_safe(tk)
        if price is None:
            price = get_history_price_fallback(tk)

        # dividends
        dividends = None
        try:
            dividends = tk.dividends
        except Exception:
            dividends = None

        # Yield: prefer info dividendYield if sane, else LTM from dividends
        y_info = normalize_percent(safe_float(info.get("dividendYield")))
        y_ltm = compute_ltm_dividend_yield(dividends, price) if dividends is not None else None
        yield_pct = y_info if (y_info is not None and y_info <= 40) else y_ltm
        yield_pct, y_flags = sanity_yield(yield_pct)

        # EPS & PE
        eps = safe_float(info.get("trailingEps"))
        pe = safe_float(info.get("trailingPE"))

        # payout ratio: prefer info payoutRatio; normalize to percent
        payout = normalize_percent(safe_float(info.get("payoutRatio")))
        # fallback: compute from annual dividend (ltm) / eps
        if payout is None and eps is not None and eps != 0 and dividends is not None and price is not None:
            ltm_div = None
            try:
                end = dividends.index.max()
                start = end - pd.Timedelta(days=365)
                ltm_div = float(dividends.loc[dividends.index > start].sum())
            except Exception:
                ltm_div = None
            if ltm_div is not None:
                payout = (ltm_div / eps) * 100.0

        payout, p_flags = sanity_payout(payout, sector)

        # dividend growth info
        years_growing = compute_years_growing(dividends) if dividends is not None else None
        div_cagr_5y = compute_div_cagr(dividends, years=5) if dividends is not None else None
        div_class = dividend_class(years_growing)

        # fair value: simple, stable approach
        # - if pe and eps are present -> fairPE = clamp(pe, 10..28) unless pe absurd
        # - else default fairPE by sector
        sector_defaults = {
            "Technology": 22,
            "Consumer Defensive": 20,
            "Consumer Cyclical": 18,
            "Healthcare": 20,
            "Industrials": 18,
            "Financial Services": 12,
            "Energy": 12,
            "Utilities": 16,
            "Real Estate": 16,
            "Communication Services": 18,
            "Basic Materials": 14,
        }
        fair_pe = None
        if pe is not None and pe > 0 and pe < 80:
            fair_pe = clamp(pe, 10, 28)
        else:
            fair_pe = float(sector_defaults.get(sector, 18))

        fair_value = None
        if eps is not None and fair_pe is not None:
            fair_value = eps * fair_pe

        upside = None
        if price is not None and price > 0 and fair_value is not None:
            upside = (fair_value / price - 1.0) * 100.0

        flags = []
        flags.extend(y_flags)
        flags.extend(p_flags)

        score, signal, conf, why = score_signal_conf(
            yield_pct=yield_pct,
            payout_pct=payout,
            upside_pct=upside,
            div_cagr_5y=div_cagr_5y,
            years_growing=years_growing,
            div_class=div_class,
            flags=flags,
            has_price=(price is not None and price > 0),
            has_name=(name != ""),
        )

        row = {
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
            "DivCAGR_5Y_%": div_cagr_5y,
            "YearsGrowing": years_growing,
            "DividendClass": div_class,
            "Score": score,
            "Signal": signal,
            "Confidence": conf,
            "Why": why,
            "Flags": " | ".join(flags) if flags else "",
        }

        return row

    except Exception as e:
        # hard fail for one ticker should never kill the run
        return {
            "GeneratedUTC": generated_utc,
            "Ticker": ticker,
            "Name": "",
            "Country": country,
            "Currency": currency_guess or "",
            "Exchange": "",
            "Sector": "",
            "Industry": "",
            "Price": "",
            "DividendYield_%": "",
            "PayoutRatio_%": "",
            "PE": "",
            "EPS": "",
            "FairPE": "",
            "FairValue": "",
            "Upside_%": "",
            "DivCAGR_5Y_%": "",
            "YearsGrowing": "",
            "DividendClass": "",
            "Score": 0,
            "Signal": "ERROR",
            "Confidence": "Low",
            "Why": f"Data error: {type(e).__name__}",
            "Flags": "Fetch failed",
        }


def format_number(v: Any, decimals: int = 2) -> str:
    f = safe_float(v)
    if f is None:
        return ""
    return f"{f:.{decimals}f}"


def main() -> None:
    tickers = read_tickers(TICKERS_FILE)
    generated_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(tickers, 1):
        r = build_row(t, generated_utc)
        # never append None
        if r is not None:
            rows.append(r)
        time.sleep(0.35)

    # Normalize output strictly to COLUMNS and format fields (keep numeric as plain, UI can format)
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = {}
        for c in COLUMNS:
            rr[c] = (r or {}).get(c, "")
        out_rows.append(rr)

    # Write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Copy to docs
    DOCS_CSV.write_text(OUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Generated {len(out_rows)} rows -> {OUT_CSV} and {DOCS_CSV}")


if __name__ == "__main__":
    main()
