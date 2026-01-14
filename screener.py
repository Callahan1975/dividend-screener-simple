import os
import csv
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


TICKERS_FILE = "tickers.txt"
OUT_CSV = "data/screener_results/screener_results.csv"


def read_tickers(path: str) -> list[str]:
    tickers: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow comma/space separated
            parts = [p.strip() for p in line.replace(",", " ").split()]
            tickers.extend([p for p in parts if p])
    # unique preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main():
    tickers = read_tickers(TICKERS_FILE)
    if not tickers:
        raise SystemExit("No tickers found in tickers.txt")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    # Chunk to avoid yfinance issues with huge batches
    chunk_size = 50

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        data = yf.Tickers(" ".join(chunk))

        for t in chunk:
            try:
                tk = data.tickers.get(t)
                if tk is None:
                    continue

                info = tk.info or {}

                name = info.get("shortName") or info.get("longName") or ""
                country = info.get("country") or ""
                sector = info.get("sector") or ""
                industry = info.get("industry") or ""
                currency = info.get("currency") or ""
                exchange = info.get("exchange") or ""

                price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
                dividend_yield = safe_float(info.get("dividendYield"))  # usually fraction (0.03 = 3%)
                payout_ratio = safe_float(info.get("payoutRatio"))      # fraction
                pe = safe_float(info.get("trailingPE") or info.get("forwardPE"))

                # Convert to percents for display
                dividend_yield_pct = (dividend_yield * 100.0) if dividend_yield is not None else None
                payout_ratio_pct = (payout_ratio * 100.0) if payout_ratio is not None else None

                rows.append({
                    "Ticker": t,
                    "Name": name,
                    "Country": country,
                    "Currency": currency,
                    "Exchange": exchange,
                    "Sector": sector,
                    "Industry": industry,
                    "Price": price,
                    "DividendYield_%": dividend_yield_pct,
                    "PayoutRatio_%": payout_ratio_pct,
                    "PE": pe,
                })
            except Exception:
                # skip ticker on any error
                continue

    df = pd.DataFrame(rows)

    # Add metadata columns at end (optional)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df.insert(0, "GeneratedUTC", generated_utc)

    # Write CSV with quoting (CRITICAL)
    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL
    )

    print(f"Saved {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
