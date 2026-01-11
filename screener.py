import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# =========================
# Config (tweak these)
# =========================
TARGET_YIELD_BUY = 2.5   # % target yield used for "Fair Value"
ENTRY_MARGIN = 0.10      # 10% discount vs Fair Value for "Entry Buy Price"

# Rating logic thresholds (simple & robust)
BUY_IF_PRICE_BELOW_ENTRY = True
HOLD_IF_PRICE_BELOW_FAIR = True

# Output paths
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "input.csv"
OUTPUT_CSV = ROOT / "output.csv"

DOCS_DIR = ROOT / "docs"
DOCS_INDEX = DOCS_DIR / "index.html"


def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Create input.csv with a 'Ticker' column.")
    df = pd.read_csv(path)
    # Accept "Ticker" or first column
    if "Ticker" in df.columns:
        tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
    else:
        tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    # Deduplicate while keeping order
    seen = set()
    out = []
    for t in tickers:
        if t and t not in seen:
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


def calc_fair_and_entry(price: float | None, div_rate: float | None):
    """
    Fair Value is based on target yield:
      Fair = DividendRate / (TargetYield/100)
      Entry = Fair * (1 - ENTRY_MARGIN)
    """
    if price in (None, 0) or div_rate in (None, 0):
        return None, None, None, None

    fair = div_rate / (TARGET_YIELD_BUY / 100.0)
    entry = fair * (1 - ENTRY_MARGIN)

    to_fair = (fair / price - 1) * 100
    to_entry = (entry / price - 1) * 100

    return round(fair, 2), round(entry, 2), round(to_fair, 1), round(to_entry, 1)


def quality_score(div_yield_pct, payout_pct, pe_ttm):
    """
    Simple scoring (0-100). You can refine later.
    Emphasis: reasonable valuation + sustainable payout + decent yield.
    """
    score = 60

    # Yield contribution
    if div_yield_pct is None:
        score -= 10
    else:
        if div_yield_pct >= 3.0:
            score += 20
        elif div_yield_pct >= 2.0:
            score += 15
        elif div_yield_pct >= 1.0:
            score += 8
        else:
            score -= 5

    # Payout sustainability
    if payout_pct is None:
        score -= 5
    else:
        if payout_pct <= 60:
            score += 15
        elif payout_pct <= 75:
            score += 5
        elif payout_pct <= 90:
            score -= 10
        else:
            score -= 20

    # Valuation
    if pe_ttm is None:
        score -= 5
    else:
        if pe_ttm <= 18:
            score += 15
        elif pe_ttm <= 25:
            score += 8
        elif pe_ttm <= 35:
            score -= 5
        else:
            score -= 15

    # Clamp
    score = max(0, min(100, score))
    return int(round(score))


def rating_from_prices(price, entry, fair):
    """
    BUY  = price <= entry
    HOLD = entry < price <= fair
    SELL = price > fair
    """
    if price is None or (entry is None and fair is None):
        return "HOLD"

    if entry is not None and BUY_IF_PRICE_BELOW_ENTRY and price <= entry:
        return "BUY"
    if fair is not None and HOLD_IF_PRICE_BELOW_FAIR and price <= fair:
        return "HOLD"
    return "SELL"


def fetch_rows(tickers: list[str]) -> list[dict]:
    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception:
            info = {}

        name = info.get("shortName") or info.get("longName") or ""
        currency = info.get("currency")
        sector = info.get("sector")
        industry = info.get("industry")

        market_cap = safe_float(info.get("marketCap"))
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

        # IMPORTANT: Use dividendRate + price to compute yield (avoids yfinance dividendYield bugs)
        div_rate = safe_float(info.get("dividendRate"))  # annual dividend in USD
        payout = safe_float(info.get("payoutRatio"))     # decimal (0.50 = 50%)
        pe_ttm = safe_float(info.get("trailingPE"))

        # Derived fields
        div_yield_pct = None
        if price not in (None, 0) and div_rate not in (None, 0):
            div_yield_pct = (div_rate / price) * 100

        payout_pct = None if payout is None else payout * 100

        fair, entry, to_fair, to_entry = calc_fair_and_entry(price, div_rate)

        score = quality_score(
            div_yield_pct=div_yield_pct,
            payout_pct=payout_pct,
            pe_ttm=pe_ttm
        )

        rating = rating_from_prices(price, entry, fair)

        rows.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Industry": industry,
            "Currency": currency,

            "Price": None if price is None else round(price, 2),
            "Market Cap (USD bn)": None if market_cap is None else round(market_cap / 1e9, 1),

            "Dividend Rate (annual)": None if div_rate is None else round(div_rate, 2),
            "Dividend Yield (%)": None if div_yield_pct is None else round(div_yield_pct, 2),
            "Payout Ratio (%)": None if payout_pct is None else round(payout_pct, 2),

            "PE (TTM)": None if pe_ttm is None else round(pe_ttm, 2),

            "Score": score,
            "Rating": rating,

            "Fair Value (Target Yield)": fair,
            "Entry Buy Price": entry,
            "To Fair (%)": to_fair,
            "To Entry (%)": to_entry,
        })

    return rows


def df_to_html(df: pd.DataFrame, updated_text: str) -> str:
    # Simple styling + colored rating badges
    css = """
    <style>
      body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px 0; }
      .sub { color: #555; margin: 0 0 18px 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
      th { background: #f3f3f3; position: sticky; top: 0; }
      td:first-child, th:first-child { text-align: left; }
      td:nth-child(2), th:nth-child(2) { text-align: left; }
      .badge { font-weight: 700; padding: 4px 10px; border-radius: 12px; display: inline-block; text-align: center; min-width: 60px; }
      .BUY  { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
      .HOLD { background: #fff8e1; color: #7a5a00; border: 1px solid #ffe082; }
      .SELL { background: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a; }
      .small { font-size: 12px; color: #666; }
    </style>
    """

    df2 = df.copy()

    # Format Rating as badge
    def fmt_rating(x):
        if pd.isna(x):
            return ""
        x = str(x).upper()
        cls = "HOLD"
        if x in ("BUY", "HOLD", "SELL"):
            cls = x
        return f'<span class="badge {cls}">{x}</span>'

    if "Rating" in df2.columns:
        df2["Rating"] = df2["Rating"].apply(fmt_rating)

    # Render HTML table (escape=False to allow badges)
    table_html = df2.to_html(index=False, escape=False)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>Dividend Screener</title>
        {css}
      </head>
      <body>
        <h1>Dividend Screener</h1>
        <p class="sub">{updated_text}</p>
        {table_html}
        <p class="small">Model: Fair Value based on target yield {TARGET_YIELD_BUY:.2f}% and Entry at {int(ENTRY_MARGIN*100)}% discount.</p>
      </body>
    </html>
    """
    return html


def main():
    tickers = read_tickers(INPUT_CSV)
    rows = fetch_rows(tickers)

    df = pd.DataFrame(rows)

    # Sort: best opportunities first (closest to / below entry), then score
    # If To Entry is negative => price below entry (good)
    if "To Entry (%)" in df.columns:
        df["__to_entry_sort"] = df["To Entry (%)"].fillna(9999)
        df.sort_values(by=["__to_entry_sort", "Score"], ascending=[True, False], inplace=True)
        df.drop(columns=["__to_entry_sort"], inplace=True)
    else:
        df.sort_values(by=["Score"], ascending=[False], inplace=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)

    # Save live HTML in docs/
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).strftime("Updated automatically via GitHub Actions — %Y-%m-%d %H:%M UTC")
    html = df_to_html(df, updated_text=updated)
    DOCS_INDEX.write_text(html, encoding="utf-8")

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {DOCS_INDEX}")


if __name__ == "__main__":
    main()
