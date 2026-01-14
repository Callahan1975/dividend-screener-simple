from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


TICKERS_FILE = Path("tickers.txt")
OUT_DIR = Path("data/screener_results")
OUT_CSV = OUT_DIR / "screener_results.csv"

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
]


def read_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow inline comments: "TICKER  # comment"
        if "#" in line:
          line = line.split("#", 1)[0].strip()
        if line:
            tickers.append(line)
    # de-dup preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def normalize_percent(value: Optional[float]) -> Optional[float]:
    """
    Normalizes numbers that might be:
      - fraction: 0.025 => 2.5
      - percent: 2.5 => 2.5
      - already *100 twice: 250 => 2.5
    """
    if value is None:
        return None

    v = float(value)

    # negative / nonsense: just return as-is (caller may blank it out)
    if v < 0:
        return v

    # typical fractions
    if 0 <= v <= 1:
        return v * 100.0

    # if it's extremely high, it's usually double-scaled
    if v > 100:
        return v / 100.0

    # 1..100 treat as percent already
    return v


def safe_round(x: Optional[float], nd: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def fetch_info(ticker: str, tries: int = 3, sleep_s: float = 1.0) -> Dict[str, Any]:
    last_err = None
    for i in range(tries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.get_info()
            if isinstance(info, dict) and info:
                return info
            # sometimes empty dict; try history fallback
            last_err = RuntimeError("Empty info dict")
        except Exception as e:
            last_err = e
        time.sleep(sleep_s * (i + 1))
    # return empty dict if fail (we keep ticker but blank values)
    return {}


def pick_price(info: Dict[str, Any]) -> Optional[float]:
    # Try common fields in order
    for k in ["currentPrice", "regularMarketPrice", "previousClose"]:
        v = _to_float(info.get(k))
        if v is not None and v > 0:
            return v
    return None


def pick_dividend_yield(info: Dict[str, Any]) -> Optional[float]:
    # yfinance may return either dividendYield or trailingAnnualDividendYield
    for k in ["dividendYield", "trailingAnnualDividendYield"]:
        v = _to_float(info.get(k))
        if v is not None:
            return normalize_percent(v)
    return None


def pick_payout_ratio(info: Dict[str, Any]) -> Optional[float]:
    v = _to_float(info.get("payoutRatio"))
    if v is None:
        return None
    return normalize_percent(v)


def pick_pe(info: Dict[str, Any]) -> Optional[float]:
    for k in ["trailingPE", "forwardPE"]:
        v = _to_float(info.get(k))
        if v is not None and v != 0:
            return v
    return None


def to_row(ticker: str, info: Dict[str, Any], now_utc: str) -> Dict[str, Any]:
    name = info.get("shortName") or info.get("longName") or ""
    country = info.get("country") or ""
    currency = info.get("currency") or ""
    exchange = info.get("exchange") or info.get("fullExchangeName") or ""
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""

    price = safe_round(pick_price(info), 2)
    div_y = safe_round(pick_dividend_yield(info), 2)
    payout = safe_round(pick_payout_ratio(info), 2)
    pe = safe_round(pick_pe(info), 2)

    # If extreme weird values still slip through, blank them out:
    # Dividend yield: if > 30% it's often junk (except special cases). You can tweak.
    if div_y is not None and div_y > 30:
        # keep but you can choose to blank. I'd rather keep but capped? Here: blank.
        div_y = None

    # Payout ratio: allow up to 200% (REITs/MLPs can be weird), else blank.
    if payout is not None and payout > 2000:
        payout = None

    row = {
        "GeneratedUTC": now_utc,
        "Ticker": ticker,
        "Name": name,
        "Country": country,
        "Currency": currency,
        "Exchange": exchange,
        "Sector": sector,
        "Industry": industry,
        "Price": price,
        "DividendYield_%": div_y,
        "PayoutRatio_%": payout,
        "PE": pe,
    }
    return row


def main() -> None:
    tickers = read_tickers(TICKERS_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat(sep=" ") + " UTC"

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        info = fetch_info(t)
        rows.append(to_row(t, info, now_utc))

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Ensure consistent CSV formatting
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
