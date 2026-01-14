from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

TICKERS_FILE = Path("tickers.txt")

# Where the script writes results (kept as you have it)
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

    # New “decision” columns
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
    yfinance can return percentages as 0.034 or 3.4 depending on field.
    We normalize to "percent points" (e.g. 3.4 means 3.4%).
    Heuristic:
      - if value <= 1.5 -> assume it's fraction (0.034)
      - else assume already percent points (3.4)
    """
    if x is None:
        return None
    if x <= 1.5:
        return x * 100.0
    return x


def fetch_info(ticker: str) -> Dict[str, Any]:
    """
    Try to fetch yfinance info robustly.
    """
    t = yf.Ticker(ticker)
    info: Dict[str, Any] = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return info


def pick_price(info: Dict[str, Any]) -> Optional[float]:
    # prefer currentPrice then regularMarketPrice
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
    # Prefer forward EPS for valuation, fallback to trailing EPS
    for k in ("forwardEps", "trailingEps"):
        v = to_float(info.get(k))
        if v and v > 0:
            return v
    return None


def pick_dividend_yield_percent(info: Dict[str, Any]) -> Optional[float]:
    """
    Prefer dividendYield from yfinance (fraction), normalize to % points.
    Fallback to trailingAnnualDividendYield, etc.
    """
    for k in ("dividendYield", "trailingAnnualDividendYield"):
        v = to_float(info.get(k))
        if v is None:
            continue
        v = normalize_percent(v)
        if v and v >= 0:
            return v
    return None


def pick_payout_ratio_percent(info: Dict[str, Any]) -> Optional[float]:
    """
    payoutRatio is typically fraction (0.55), normalize to % points.
    """
    v = to_float(info.get("payoutRatio"))
    if v is None:
        return None
    v = normalize_percent(v)
    if v >= 0:
        return v
    return None


def sector_is_reit_like(sector: str, industry: str) -> bool:
    s = (sector or "").lower()
    i = (industry or "").lower()
    return ("real estate" in s) or ("reit" in i) or ("reit" in s)


def compute_fair_pe(pe: Optional[float], sector: str) -> Optional[float]:
    """
    Very stable “fair PE” heuristic:
      - If we have PE: clamp it lightly to avoid crazy fair values
      - If missing: use sector defaults (broad, stable)
    """
    sector_l = (sector or "").lower()

    # sector fallback defaults (rough but stable)
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
        # “fair” is not equal to current PE, but we can dampen extremes:
        # clamp current PE into a reasonable band as a proxy
        return clamp(pe, 8.0, 28.0)

    # fallback by sector keyword
    for key, val in defaults.items():
        if key in sector_l:
            return val
    return 18.0  # generic


def compute_upside(price: Optional[float], eps: Optional[float], fair_pe: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (fair_value, upside_%)
    upside_% clamped to [-50, +100]
    """
    if not price or price <= 0:
        return (None, None)
    if not eps or eps <= 0:
        return (None, None)
    if not fair_pe or fair_pe <= 0:
        return (None, None)

    fair_value = eps * fair_pe
    upside = (fair_value / price - 1.0) * 100.0
    upside = clamp(upside, -50.0, 100.0)
    return (fair_value, upside)


def compute_score_signal_confidence(
    sector: str,
    industry: str,
    yield_pct: Optional[float],
    payout_pct: Optional[float],
    pe: Optional[float],
    upside_pct: Optional[float],
    eps: Optional[float],
    price: Optional[float],
) -> Tuple[int, str, str, str]:
    """
    Simple “decision grade” model:
      - Score 0..100
      - Signal GOLD/BUY/HOLD/WATCH
      - Confidence High/Med/Low
      - Why (short)
    """

    flags: List[str] = []
    score = 50  # baseline

    # --- Data quality / confidence base ---
    conf = 100
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

    # --- Yield sanity ---
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

    # --- Payout sanity (sector-aware-ish) ---
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

    # --- Valuation via Upside ---
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

    # --- PE sanity (avoid cheap traps / crazy values) ---
    if pe is not None:
        if pe > 40:
            score -= 6
            flags.append("PE very high")
        elif pe < 6:
            score -= 4
            flags.append("PE very low")

    # clamp score
    score = int(max(0, min(100, score)))

    # confidence label
    if conf >= 80:
        confidence = "High"
    elif conf >= 60:
        confidence = "Med"
    else:
        confidence = "Low"

    # signal
    # If confidence low, don’t allow GOLD/BUY
    if confidence == "Low":
        signal = "WATCH"
    else:
        if score >= 85 and (upside_pct is not None and upside_pct >= 10) and ("Payout very high" not in flags):
            signal = "GOLD"
        elif score >= 75 and (upside_pct is not None and upside_pct >= 5):
            signal = "BUY"
        elif score >= 60:
            signal = "HOLD"
        else:
            signal = "WATCH"

    # short why (pick the most useful 1-2)
    why_parts: List[str] = []
    # prioritize warnings
    for p in ("Yield outlier", "Payout very high", "Payout high", "Overvalued", "Missing EPS", "No upside calc"):
        if p in flags:
            why_parts.append(p)
            break
    # then add a positive if any
    for p in ("Undervalued", "Good upside", "High yield"):
        if p in flags and p not in why_parts:
            why_parts.append(p)
            break
    why = " / ".join(why_parts) if why_parts else ""

    return score, signal, confidence, why


def to_row(ticker: str, info: Dict[str, Any]) -> Dict[str, Any]:
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

    div_yield_pct = pick_dividend_yield_percent(info)
    payout_pct = pick_payout_ratio_percent(info)

    # Fair value + upside
    fair_pe = compute_fair_pe(pe, sector)
    fair_value, upside_pct = compute_upside(price, eps, fair_pe)

    # Score / Signal / Confidence / Why
    score, signal, confidence, why = compute_score_signal_confidence(
        sector=sector,
        industry=industry,
        yield_pct=div_yield_pct,
        payout_pct=payout_pct,
        pe=pe,
        upside_pct=upside_pct,
        eps=eps,
        price=price,
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

        "FairPE": safe_round(fair_pe, 1),
        "FairValue": safe_round(fair_value, 2),
        "Upside_%": safe_round(upside_pct, 1),
        "Score": score,
        "Signal": signal,
        "Confidence": confidence,
        "Why": why,
    }

    # ensure all columns exist (no DataTables issues)
    for c in COLUMNS:
        row.setdefault(c, "")

    return row


def main() -> None:
    tickers = read_tickers(TICKERS_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(tickers, start=1):
        try:
            info = fetch_info(t)
            row = to_row(t, info)
            rows.append(row)
        except Exception as e:
            # still write a row so UI doesn't break
            rows.append({c: "" for c in COLUMNS} | {
                "GeneratedUTC": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": t,
                "Name": "",
                "Why": f"Error: {type(e).__name__}",
                "Signal": "WATCH",
                "Confidence": "Low",
                "Score": 0,
            })

        # be nice to API
        time.sleep(0.4)

    # write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            # DictWriter requires keys exist
            out = {c: r.get(c, "") for c in COLUMNS}
            w.writerow(out)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
