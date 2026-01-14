import csv
import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional

import yfinance as yf


TICKERS_FILE = "tickers.txt"
OUT_DIR = os.path.join("data", "screener_results")
OUT_CSV = os.path.join(OUT_DIR, "screener_results.csv")


def read_tickers(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # allow inline comments
            if "#" in s:
                s = s.split("#", 1)[0].strip()
            if s:
                tickers.append(s)
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def normalize_percent(value: Optional[float]) -> Optional[float]:
    """
    Normalize percent-like metrics so we store:
      - DividendYield_% as 0..100 (e.g., 3.04)
      - PayoutRatio_%  as 0..500-ish (some can be >100)

    Rules:
      - if value is 0..1 => treat as fraction and multiply by 100
      - if value is 1..100 => treat as already percent
      - if value is huge (>5000) => probably wrong, return None
    """
    if value is None:
        return None

    # Sometimes yfinance gives 0.0304 (fraction)
    if 0 <= value <= 1:
        return value * 100

    # Sometimes we already computed percent (e.g. 3.04)
    if 1 < value <= 1000:
        return value

    # absurd
    if value > 5000:
        return None

    return value


def get_info(ticker: str, pause: float = 0.2) -> Dict[str, Any]:
    """
    Robust yfinance fetch: use fast_info where possible and info as fallback.
    """
    t = yf.Ticker(ticker)

    info: Dict[str, Any] = {}
    try:
        # info can be slow / rate limited; still ok in moderation
        info = t.info or {}
    except Exception:
        info = {}

    # fast_info often more reliable for price
    fast = {}
    try:
        fast = getattr(t, "fast_info", {}) or {}
    except Exception:
        fast = {}

    # small pause to reduce throttling
    time.sleep(pause)

    return {"info": info, "fast": fast}


def pick_first(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tickers = read_tickers(TICKERS_FILE)
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    rows: List[Dict[str, Any]] = []

    for tk in tickers:
        try:
            data = get_info(tk)
            info = data["info"]
            fast = data["fast"]

            name = pick_first(info.get("shortName"), info.get("longName"), "")
            country = pick_first(info.get("country"), "")
            currency = pick_first(info.get("currency"), "")
            exchange = pick_first(info.get("exchange"), info.get("exchangeName"), "")
            sector = pick_first(info.get("sector"), "")
            industry = pick_first(info.get("industry"), "")

            # Price
            price = safe_float(
                pick_first(
                    fast.get("last_price"),
                    fast.get("lastPrice"),
                    info.get("currentPrice"),
                    info.get("regularMarketPrice"),
                )
            )

            # Dividend yield (yfinance often provides dividendYield as fraction 0.03)
            div_yield_raw = safe_float(info.get("dividendYield"))

            # If dividendYield is missing, we can compute from trailing annual dividend if present
            # (NOTE: info.get("dividendRate") is typically annual dividend per share)
            div_rate = safe_float(info.get("dividendRate"))
            div_yield_calc = None
            if (div_yield_raw is None or div_yield_raw == 0) and price and div_rate:
                div_yield_calc = (div_rate / price) * 100  # already percent

            dividend_yield_pct = normalize_percent(
                pick_first(div_yield_raw, div_yield_calc)
            )

            # Payout ratio (often fraction 0.55)
            payout_raw = safe_float(info.get("payoutRatio"))
            payout_pct = normalize_percent(payout_raw)

            # PE
            pe = safe_float(
                pick_first(
                    info.get("trailingPE"),
                    info.get("forwardPE"),
                )
            )

            rows.append(
                {
                    "GeneratedUTC": now_utc,
                    "Ticker": tk,
                    "Name": name,
                    "Country": country,
                    "Currency": currency,
                    "Exchange": exchange,
                    "Sector": sector,
                    "Industry": industry,
                    "Price": price,
                    "DividendYield_%": dividend_yield_pct,
                    "PayoutRatio_%": payout_pct,
                    "PE": pe,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "GeneratedUTC": now_utc,
                    "Ticker": tk,
                    "Name": "",
                    "Country": "",
                    "Currency": "",
                    "Exchange": "",
                    "Sector": "",
                    "Industry": "",
                    "Price": "",
                    "DividendYield_%": "",
                    "PayoutRatio_%": "",
                    "PE": "",
                    "Error": str(e),
                }
            )

    # Write CSV
    fieldnames = [
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

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            clean = {k: r.get(k, "") for k in fieldnames}
            w.writerow(clean)

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
