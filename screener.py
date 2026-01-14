import os
import time
import csv
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


TICKERS_FILE = "tickers.txt"
OUT_DIR = os.path.join("data", "screener_results")
OUT_CSV = os.path.join(OUT_DIR, "screener_results.csv")


def load_tickers(path: str) -> list[str]:
    tickers: list[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty lines and full-line comments
            if not line or line.startswith("#"):
                continue

            # allow inline comments: "ABC # comment"
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if line:
                tickers.append(line)

    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_get(d: dict, key: str, default=None):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tickers = load_tickers(TICKERS_FILE)
    if not tickers:
        raise RuntimeError("No tickers found in tickers.txt")

    rows = []
    errors = []

    # Slight throttling helps avoid sporadic Yahoo rate limiting
    for i, t in enumerate(tickers, start=1):
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}

            rows.append(
                {
                    "Ticker": t,
                    "Name": safe_get(info, "longName", safe_get(info, "shortName", "")),
                    "Country": safe_get(info, "country", ""),
                    "Currency": safe_get(info, "currency", ""),
                    "Exchange": safe_get(info, "exchange", ""),
                    "Sector": safe_get(info, "sector", ""),
                    "Industry": safe_get(info, "industry", ""),
                    "Price": safe_get(info, "regularMarketPrice", safe_get(info, "currentPrice", "")),
                    "MarketCap": safe_get(info, "marketCap", ""),
                    "DividendYield_%": (
                        float(info["dividendYield"]) * 100.0
                        if isinstance(info.get("dividendYield", None), (int, float))
                        else ""
                    ),
                    "PayoutRatio": safe_get(info, "payoutRatio", ""),
                    "PE": safe_get(info, "trailingPE", safe_get(info, "forwardPE", "")),
                }
            )

        except Exception as e:
            errors.append({"Ticker": t, "Error": str(e)})

        # throttle a bit
        time.sleep(0.25)

    df = pd.DataFrame(rows)

    # Sort for stable output
    if "Country" in df.columns and "Ticker" in df.columns:
        df = df.sort_values(["Country", "Ticker"], kind="stable")

    # Write CSV with correct quoting so names like "T-Mobile US, Inc." won't break
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df.insert(0, "GeneratedUTC", generated_utc)

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    # Optional: write an errors file for debugging
    if errors:
        err_path = os.path.join(OUT_DIR, "errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    print(f"Generated rows: {len(df)}")
    print(f"CSV written to: {OUT_CSV}")
    if errors:
        print(f"Errors: {len(errors)} (see {os.path.join(OUT_DIR, 'errors.csv')})")


if __name__ == "__main__":
    main()
