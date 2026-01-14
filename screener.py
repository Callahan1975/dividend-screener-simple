import csv
import datetime as dt
import time
from pathlib import Path

import yfinance as yf


TICKERS_FILE = Path("tickers.txt")

OUT_DATA_DIR = Path("data") / "screener_results"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DATA_DIR / "screener_results.csv"


# Output columns (keep stable so UI doesn't break)
FIELDS = [
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


def read_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Skip obviously invalid tickers
        if "?" in s:
            continue
        tickers.append(s)
    # de-dup preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def to_pct(value) -> float | None:
    """
    yfinance sometimes returns:
      - 0.0312 (fraction)
      - 3.12 (already percent)
      - None
    We normalize to percent (e.g. 3.12)
    """
    v = safe_float(value)
    if v is None:
        return None
    # If it looks like a fraction (0-1.5), convert to percent
    if 0 <= v <= 1.5:
        return v * 100.0
    # If it is already percent-like (1.5-200), keep
    if 1.5 < v <= 200:
        return v
    # If it's crazy huge (e.g. 304), it might be percent*100
    if v > 200:
        # Try divide by 100
        v2 = v / 100.0
        if 0 <= v2 <= 200:
            return v2
    return v


def pick_first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] not in (None, "", "N/A"):
            return d[k]
    return None


def get_last_12m_div_yield_pct(ticker_obj: yf.Ticker, price: float | None) -> float | None:
    """
    Fallback: sum dividends last 365 days / price * 100
    """
    if price is None or price <= 0:
        return None
    try:
        divs = ticker_obj.dividends
        if divs is None or len(divs) == 0:
            return None
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=365)
        divs_12m = divs[divs.index.to_pydatetime() >= cutoff]
        if len(divs_12m) == 0:
            return None
        total = float(divs_12m.sum())
        return (total / price) * 100.0
    except Exception:
        return None


def fetch_one(ticker: str, pause_s: float = 0.25) -> dict:
    generated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    row = {k: "" for k in FIELDS}
    row["GeneratedUTC"] = generated
    row["Ticker"] = ticker

    t = yf.Ticker(ticker)

    # fast_info is quicker/less fragile when it works
    price = None
    currency = None
    exchange = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            price = safe_float(fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice"))
            currency = fi.get("currency")
            exchange = fi.get("exchange")
    except Exception:
        pass

    # main info (can be slower)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    name = pick_first(info, ["longName", "shortName", "displayName", "name"])
    country = pick_first(info, ["country"])
    sector = pick_first(info, ["sector"])
    industry = pick_first(info, ["industry"])
    currency = currency or pick_first(info, ["currency"])
    exchange = exchange or pick_first(info, ["exchange", "fullExchangeName"])

    # price fallback from info
    if price is None:
        price = safe_float(pick_first(info, ["regularMarketPrice", "currentPrice", "previousClose"]))

    # dividend yield + payout + PE
    dy_raw = pick_first(
        info,
        [
            "dividendYield",                 # often fraction
            "trailingAnnualDividendYield",   # often fraction
        ],
    )
    dy_pct = to_pct(dy_raw)

    # fallback from dividends series if missing
    if dy_pct is None:
        dy_pct = get_last_12m_div_yield_pct(t, price)

    payout_raw = pick_first(info, ["payoutRatio"])
    payout_pct = to_pct(payout_raw)

    pe = safe_float(pick_first(info, ["trailingPE", "forwardPE"]))

    # write
    row["Name"] = name or ""
    row["Country"] = country or ""
    row["Currency"] = currency or ""
    row["Exchange"] = exchange or ""
    row["Sector"] = sector or ""
    row["Industry"] = industry or ""
    row["Price"] = f"{price:.4f}" if isinstance(price, (int, float)) and price is not None else ""
    row["DividendYield_%"] = f"{dy_pct:.4f}" if isinstance(dy_pct, (int, float)) and dy_pct is not None else ""
    row["PayoutRatio_%"] = f"{payout_pct:.4f}" if isinstance(payout_pct, (int, float)) and payout_pct is not None else ""
    row["PE"] = f"{pe:.4f}" if isinstance(pe, (int, float)) and pe is not None else ""

    time.sleep(pause_s)
    return row


def main():
    tickers = read_tickers(TICKERS_FILE)
    if not tickers:
        raise SystemExit("No tickers found in tickers.txt")

    rows = []
    for i, tk in enumerate(tickers, 1):
        try:
            rows.append(fetch_one(tk))
            if i % 25 == 0:
                print(f"Fetched {i}/{len(tickers)}")
        except Exception as e:
            print(f"[WARN] {tk} failed: {e}")

    # Write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
