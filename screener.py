#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) with Portfolio Integration (Snowball CSV)

Outputs:
- data/screener_results.csv
- data/screener_results_portfolio.csv
- docs/index.html

Requires:
- yfinance, pandas, numpy, lxml, html5lib, requests, pyyaml
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from src.portfolio_ingest import load_positions_from_snowball
from src.portfolio_actions import apply_portfolio_actions, load_rules


ROOT = Path(__file__).resolve().parent

TICKERS_FILE = ROOT / "tickers.txt"
OUT_DIR = ROOT / "data"
OUT_CSV = OUT_DIR / "screener_results.csv"
OUT_CSV_PORTFOLIO = OUT_DIR / "screener_results_portfolio.csv"
DOCS_DIR = ROOT / "docs"
OUT_HTML = DOCS_DIR / "index.html"

SNOWBALL_PATH = ROOT / "data" / "portfolio" / "Snowball.csv"
ALIAS_PATH = ROOT / "data" / "portfolio" / "ticker_alias.csv"
RULES_PATH = ROOT / "config" / "portfolio_rules.yml"


def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow comments at end of line
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if not s:
            continue
        tickers.append(s)
    # unique, keep order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if math.isnan(v):
                return None
            return v
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def pct(x: float | None) -> float | None:
    if x is None:
        return None
    return x * 100.0


def cagr_from_dividends(div_series: pd.Series, years: int = 5) -> float | None:
    """
    Estimate dividend growth using annual sums.
    """
    try:
        if div_series is None or div_series.empty:
            return None
        s = div_series.dropna()
        if s.empty:
            return None
        # annualize
        annual = s.resample("Y").sum()
        annual = annual[annual > 0]
        if len(annual) < years + 1:
            return None
        end_year = annual.index.max()
        start_year = end_year - pd.DateOffset(years=years)
        a0 = annual[annual.index <= start_year].iloc[-1]
        a1 = annual[annual.index <= end_year].iloc[-1]
        a0 = float(a0)
        a1 = float(a1)
        if a0 <= 0 or a1 <= 0:
            return None
        return (a1 / a0) ** (1.0 / years) - 1.0
    except Exception:
        return None


def fair_value_yield(price: float | None, current_yield: float | None, target_yield: float | None) -> float | None:
    if price is None or current_yield is None or target_yield is None:
        return None
    if target_yield <= 0:
        return None
    # if yield = dividend / price  => dividend = yield * price
    dividend = current_yield * price
    return dividend / target_yield


def fair_value_ddm(dividend: float | None, growth: float | None, discount: float = 0.09) -> float | None:
    """
    Gordon Growth DDM: P = D1 / (r - g)
    dividend: trailing annual dividend (D0)
    growth: expected growth (g)
    """
    if dividend is None or growth is None:
        return None
    g = growth
    r = discount
    if dividend <= 0:
        return None
    # guardrails
    g = max(min(g, 0.12), -0.05)
    if r <= g + 0.01:
        return None
    d1 = dividend * (1.0 + g)
    return d1 / (r - g)


def upside_pct(price: float | None, fair: float | None) -> float | None:
    if price is None or fair is None:
        return None
    if price <= 0:
        return None
    return (fair / price) - 1.0


def score_row(yld: float | None, growth: float | None, pe: float | None, payout: float | None) -> float:
    """
    Simple heuristic score 0..100.
    """
    score = 50.0

    # yield contribution (cap at 7%)
    if yld is not None:
        y = min(max(yld, 0.0), 0.07)
        score += (y / 0.07) * 15.0

    # growth contribution (cap at 12%)
    if growth is not None:
        g = min(max(growth, -0.03), 0.12)
        score += (g / 0.12) * 20.0

    # PE contribution: best around 10-18, penalty above 25
    if pe is not None and pe > 0:
        if pe < 10:
            score += 6.0
        elif pe <= 18:
            score += 12.0
        elif pe <= 25:
            score += 6.0
        else:
            score -= min((pe - 25) * 1.0, 15.0)

    # payout ratio: ideal 25-65%
    if payout is not None and payout > 0:
        if payout < 0.25:
            score += 4.0
        elif payout <= 0.65:
            score += 10.0
        elif payout <= 0.90:
            score -= 3.0
        else:
            score -= 12.0

    return float(min(max(score, 0.0), 100.0))


def reco_label(score: float, upside: float | None) -> str:
    if upside is None:
        upside = 0.0
    if score >= 75 and upside >= 0.10:
        return "Strong"
    if score >= 65 and upside >= 0.05:
        return "Good"
    if score >= 55:
        return "Ok"
    return "Weak"


def action_label(score: float, upside: float | None) -> str:
    """
    Baseline action (without portfolio context).
    """
    if upside is None:
        upside = 0.0
    if score >= 75 and upside >= 0.10:
        return "BUY"
    if score >= 65 and upside >= 0.05:
        return "ADD"
    if score >= 50:
        return "HOLD"
    if score >= 40:
        return "TRIM"
    return "AVOID"


def fetch_one(ticker: str) -> dict:
    """
    Pull snapshot + dividend history via yfinance.
    Never throws: returns dict with None on missing fields.
    """
    row = {
        "Ticker": ticker,
        "Name": None,
        "Currency": None,
        "Price": None,
        "MarketCap": None,
        "PE": None,
        "PayoutRatio": None,
        "DividendYield": None,        # decimal (0.03)
        "DividendRate": None,         # trailing annual dividend
        "DividendGrowth5Y": None,     # decimal CAGR
        "TargetYield": None,          # model yield (for fair value)
        "FairValue_Yield": None,
        "FairValue_DDM": None,
        "Upside_Yield_%": None,       # decimal
        "Upside_DDM_%": None,         # decimal
        "Score": None,
        "Reco": None,
        "Action": None,
        "LastUpdatedUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "YF_Status": "OK",
    }

    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}

        # price
        price = safe_float(info.get("regularMarketPrice"))
        if price is None:
            # fallback: history
            try:
                hist = t.history(period="5d")
                if not hist.empty:
                    price = safe_float(hist["Close"].iloc[-1])
            except Exception:
                pass

        row["Price"] = price
        row["Name"] = info.get("shortName") or info.get("longName")
        row["Currency"] = info.get("currency")

        row["MarketCap"] = safe_float(info.get("marketCap"))
        row["PE"] = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        row["PayoutRatio"] = safe_float(info.get("payoutRatio"))

        dy = safe_float(info.get("dividendYield"))
        # yfinance sometimes returns percent (e.g. 0.03) which is correct; keep as decimal
        row["DividendYield"] = dy
        row["DividendRate"] = safe_float(info.get("dividendRate"))

        # dividend growth from history
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                divs.index = pd.to_datetime(divs.index)
                g5 = cagr_from_dividends(divs, years=5)
                row["DividendGrowth5Y"] = g5
        except Exception:
            pass

        # model target yield:
        # simple: use max(current_yield, 2%) and cap at 6.5% for quality
        if dy is not None:
            target = float(min(max(dy, 0.02), 0.065))
        else:
            target = 0.03
        row["TargetYield"] = target

        # fair values
        fv_y = fair_value_yield(price, dy, target)
        row["FairValue_Yield"] = fv_y

        fv_ddm = fair_value_ddm(row["DividendRate"], row["DividendGrowth5Y"], discount=0.09)
        row["FairValue_DDM"] = fv_ddm

        up_y = upside_pct(price, fv_y)
        up_d = upside_pct(price, fv_ddm)

        row["Upside_Yield_%"] = up_y
        row["Upside_DDM_%"] = up_d

        score = score_row(dy, row["DividendGrowth5Y"], row["PE"], row["PayoutRatio"])
        row["Score"] = score
        row["Reco"] = reco_label(score, up_y)
        row["Action"] = action_label(score, up_y)

        return row

    except Exception as e:
        row["YF_Status"] = f"ERROR: {type(e).__name__}"
        return row


def format_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x*100:.1f}%"


def format_money(x: float | None) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return ""


def build_html(df: pd.DataFrame) -> str:
    # Minimal self-contained HTML table
    cols = [
        "Ticker", "Name", "Price",
        "DividendYield", "DividendGrowth5Y", "PayoutRatio", "PE",
        "FairValue_Yield", "Upside_Yield_%", "Score", "Reco", "Action",
        "OwnedValue", "Weight", "PortfolioAction",
    ]
    show = [c for c in cols if c in df.columns]

    d = df.copy()

    if "DividendYield" in d.columns:
        d["DividendYield"] = d["DividendYield"].apply(format_pct)
    if "DividendGrowth5Y" in d.columns:
        d["DividendGrowth5Y"] = d["DividendGrowth5Y"].apply(format_pct)
    if "PayoutRatio" in d.columns:
        d["PayoutRatio"] = d["PayoutRatio"].apply(format_pct)
    if "Upside_Yield_%" in d.columns:
        d["Upside_Yield_%"] = d["Upside_Yield_%"].apply(format_pct)
    if "Weight" in d.columns:
        d["Weight"] = d["Weight"].apply(format_pct)

    for m in ["Price", "FairValue_Yield", "OwnedValue"]:
        if m in d.columns:
            d[m] = d[m].apply(format_money)

    if "Score" in d.columns:
        d["Score"] = d["Score"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")

    table_html = d[show].to_html(index=False, escape=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Dividend Screener (DK + US)</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 18px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .meta {{ color: #555; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .hint {{ color: #666; font-size: 12px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>Dividend Screener (DK + US)</h1>
  <div class="meta">Updated: {ts}</div>
  {table_html}
  <div class="hint">
    Columns: Action = baseline screener. PortfolioAction = ADD/BUY/HOLD/TRIM/AVOID based on Snowball portfolio + rules.
  </div>
</body>
</html>
"""


def run_screener() -> pd.DataFrame:
    tickers = read_tickers(TICKERS_FILE)

    rows = []
    for tk in tickers:
        r = fetch_one(tk)
        rows.append(r)

    df = pd.DataFrame(rows)

    # sort: best first (Score desc, Upside desc)
    if "Score" in df.columns and "Upside_Yield_%" in df.columns:
        df = df.sort_values(["Score", "Upside_Yield_%"], ascending=[False, False], na_position="last")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    return df


def main():
    df = run_screener()

    # portfolio integration (safe even if files missing)
    positions = load_positions_from_snowball(
        snowball_csv_path=SNOWBALL_PATH,
        alias_csv_path=ALIAS_PATH,
    )
    rules = load_rules(RULES_PATH)

    dfp = apply_portfolio_actions(df, positions, rules)
    dfp.to_csv(OUT_CSV_PORTFOLIO, index=False, encoding="utf-8")

    html = build_html(dfp)
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_CSV_PORTFOLIO}")
    print(f"Wrote: {OUT_HTML}")


if __name__ == "__main__":
    main()
