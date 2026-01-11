from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf


# -----------------------
# Settings (tweakable)
# -----------------------
DEFAULT_TARGET_YIELD_PCT = 2.75   # fallback target yield if 5y avg is missing
ENTRY_DISCOUNT_PCT = 10.0         # extra discount on the yield-entry price
HOLD_BAND_PCT = 10.0              # if price is within 10% above entry => HOLD, else SELL


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "input.csv"
OUTPUT_CSV = ROOT / "output.csv"
DOCS_DIR = ROOT / "docs"
DOCS_INDEX = DOCS_DIR / "index.html"
DOCS_OUTPUT_CSV = DOCS_DIR / "output.csv"


def _safe_pct(x):
    """Convert ratios to percent safely."""
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    # If already looks like percent (e.g. 49.0), keep it
    if x > 1.5:
        return round(x, 2)
    return round(x * 100.0, 2)


def _safe_float(x, nd=2):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _now_utc_str():
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def load_tickers(input_csv: Path) -> list[str]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing {input_csv}. Create it with a column named 'ticker'.")

    df = pd.read_csv(input_csv)
    cols = [c.lower().strip() for c in df.columns]
    if "ticker" not in cols:
        raise ValueError("input.csv must contain a column named 'ticker'")

    # Map real column name
    ticker_col = df.columns[cols.index("ticker")]

    tickers = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": None})
        .dropna()
        .tolist()
    )

    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def compute_row(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    name = info.get("shortName") or info.get("longName")
    sector = info.get("sector")
    industry = info.get("industry")
    currency = info.get("currency")

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    price = _safe_float(price, 2)

    market_cap = info.get("marketCap")
    market_cap_bn = None
    if market_cap is not None:
        try:
            market_cap_bn = round(float(market_cap) / 1e9, 1)
        except Exception:
            market_cap_bn = None

    # Dividend rate is annual cash dividend (e.g. 5.20 USD/year)
    dividend_rate = info.get("dividendRate")
    dividend_rate = _safe_float(dividend_rate, 2)

    # IMPORTANT:
    # yfinance's "dividendYield" can be unreliable/None/format-shifted.
    # So we compute yield ourselves: (annual dividend / current price) * 100
    dividend_yield_pct = None
    if dividend_rate is not None and price and price > 0:
        dividend_yield_pct = round((dividend_rate / price) * 100.0, 2)

    payout_ratio_pct = _safe_pct(info.get("payoutRatio"))

    pe_ttm = _safe_float(info.get("trailingPE"), 2)
    forward_pe = _safe_float(info.get("forwardPE"), 2)

    # 5y avg dividend yield from Yahoo (if present) is typically already in percent (e.g. 2.64)
    fivey_avg_yield_pct = info.get("fiveYearAvgDividendYield")
    fivey_avg_yield_pct = _safe_float(fivey_avg_yield_pct, 2)

    target_yield_pct = fivey_avg_yield_pct if fivey_avg_yield_pct is not None else DEFAULT_TARGET_YIELD_PCT
    target_yield_dec = target_yield_pct / 100.0

    # Yield-based ENTRY price (NOT intrinsic value)
    entry_price = None
    if dividend_rate is not None and target_yield_dec > 0:
        entry_price = dividend_rate / target_yield_dec
        entry_price *= (1.0 - ENTRY_DISCOUNT_PCT / 100.0)
        entry_price = round(entry_price, 2)

    to_entry_pct = None
    if price and entry_price and price > 0:
        to_entry_pct = round((entry_price / price - 1.0) * 100.0, 2)

    # Simple rating:
    rating = "N/A"
    if price and entry_price:
        if price <= entry_price:
            rating = "BUY"
        elif to_entry_pct is not None and to_entry_pct >= -HOLD_BAND_PCT:
            rating = "HOLD"
        else:
            rating = "SELL"

    # Simple score (0-100) - you can change weights later
    # Goal: reward reasonable payout + decent yield + not-crazy PE
    score = 50
    if dividend_yield_pct is not None:
        if dividend_yield_pct >= 3.0:
            score += 15
        elif dividend_yield_pct >= 2.0:
            score += 10
        elif dividend_yield_pct >= 1.0:
            score += 5

    if payout_ratio_pct is not None:
        if 0 < payout_ratio_pct <= 60:
            score += 15
        elif 60 < payout_ratio_pct <= 80:
            score += 8
        elif payout_ratio_pct > 100:
            score -= 15

    if pe_ttm is not None:
        if pe_ttm <= 18:
            score += 10
        elif pe_ttm <= 25:
            score += 5
        elif pe_ttm >= 35:
            score -= 10

    score = max(0, min(100, int(score)))

    return {
        "Ticker": ticker,
        "Name": name,
        "Sector": sector,
        "Industry": industry,
        "Currency": currency,
        "Price": price,
        "Market Cap (USD bn)": market_cap_bn,
        "Dividend Rate (annual)": dividend_rate,
        "Dividend Yield (%)": dividend_yield_pct,
        "Payout Ratio (%)": payout_ratio_pct,
        "PE (TTM)": pe_ttm,
        "Forward PE": forward_pe,
        "Target Yield (%)": round(target_yield_pct, 2) if target_yield_pct is not None else None,
        "Yield Entry Price": entry_price,
        "To Yield Entry (%)": to_entry_pct,
        "Score": score,
        "Rating": rating,
        "Updated (UTC)": _now_utc_str(),
    }


def build_html(df: pd.DataFrame) -> str:
    # Format for display
    display_df = df.copy()

    # Better column order (adjust if you want)
    preferred = [
        "Ticker", "Name", "Sector", "Industry", "Currency",
        "Price", "Market Cap (USD bn)",
        "Dividend Rate (annual)", "Dividend Yield (%)", "Payout Ratio (%)",
        "PE (TTM)", "Forward PE",
        "Target Yield (%)", "Yield Entry Price", "To Yield Entry (%)",
        "Score", "Rating",
        "Updated (UTC)"
    ]
    cols = [c for c in preferred if c in display_df.columns] + [c for c in display_df.columns if c not in preferred]
    display_df = display_df[cols]

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:,.2f}"
        return str(x)

    table_html = display_df.applymap(fmt).to_html(index=False, escape=True)

    # Minimal CSS + rating badges
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dividend Screener</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f4f4; position: sticky; top: 0; z-index: 1; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .badge {{ padding: 4px 10px; border-radius: 999px; font-weight: 600; display: inline-block; }}
    .BUY {{ background: #e7f6ea; border: 1px solid #7ed38a; color: #1f6b2a; }}
    .HOLD {{ background: #fff6e5; border: 1px solid #f3c26b; color: #7a4c00; }}
    .SELL {{ background: #ffe8e8; border: 1px solid #f08a8a; color: #8a1f1f; }}
    .wrap {{ overflow-x: auto; }}
    .note {{ margin-top: 10px; color:#666; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Dividend Screener</h1>
  <div class="sub">Updated automatically via GitHub Actions • {_now_utc_str()}</div>

  <div class="wrap">
    {table_html}
  </div>

  <div class="note">
    Yield is computed as (Dividend Rate / Price) to avoid Yahoo/yfinance inconsistencies.
    “Yield Entry Price” is an entry estimate based on target yield (not intrinsic value).
  </div>

  <script>
    // Convert Rating text to badges
    document.querySelectorAll('table tbody tr').forEach(tr => {{
      const tds = tr.querySelectorAll('td');
      if (!tds.length) return;

      // Find "Rating" column by header
      const headers = Array.from(document.querySelectorAll('table thead th')).map(h => h.innerText.trim());
      const ratingIdx = headers.indexOf('Rating');
      if (ratingIdx >= 0) {{
        const cell = tds[ratingIdx];
        const v = (cell.innerText || '').trim();
        if (v === 'BUY' || v === 'HOLD' || v === 'SELL') {{
          cell.innerHTML = `<span class="badge ${{v}}">${{v}}</span>`;
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    return html


def main():
    DOCS_DIR.mkdir(exist_ok=True)

    tickers = load_tickers(INPUT_CSV)
    if not tickers:
        raise ValueError("No tickers found in input.csv")

    rows = []
    errors = []
    for ticker in tickers:
        try:
            rows.append(compute_row(ticker))
        except Exception as e:
            errors.append((ticker, str(e)))

    df = pd.DataFrame(rows)

    # Sort: best rating first, then score desc
    rating_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "N/A": 3}
    df["_rating_sort"] = df["Rating"].map(lambda x: rating_order.get(x, 9))
    df = df.sort_values(by=["_rating_sort", "Score"], ascending=[True, False]).drop(columns=["_rating_sort"])

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_csv(DOCS_OUTPUT_CSV, index=False)

    html = build_html(df)
    DOCS_INDEX.write_text(html, encoding="utf-8")

    if errors:
        # Write errors to docs too
        err_df = pd.DataFrame(errors, columns=["Ticker", "Error"])
        (DOCS_DIR / "errors.csv").write_text(err_df.to_csv(index=False), encoding="utf-8")

        # Still exit success so you get partial results
        print("Completed with some errors:")
        for t, msg in errors:
            print(f" - {t}: {msg}")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {DOCS_INDEX}")


if __name__ == "__main__":
    main()
