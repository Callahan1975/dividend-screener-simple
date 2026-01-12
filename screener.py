#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) — Stable, portfolio-aware, with header dropdown filters.

Outputs:
- data/screener_results.csv
- docs/index.html   (DataTables with dropdown filters in column headers)

Key fixes:
- DividendYield is calculated as Trailing-12M dividends / Price (not yfinance info["dividendYield"])
- Adds filterable columns: Country, Currency, Sector, InIndex, Owned (+ portfolio columns if Snowball present)
"""

from __future__ import annotations

import os
import re
import math
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import yfinance as yf


# -----------------------
# Paths / files
# -----------------------
BASE = Path(__file__).resolve().parent

TICKERS_TXT = BASE / "tickers.txt"
INDEX_MAP_CSV = BASE / "index_map.csv"
OUT_CSV = BASE / "data" / "screener_results.csv"
OUT_HTML = BASE / "docs" / "index.html"

# Optional portfolio inputs
SNOWBALL_CSV = BASE / "data" / "portfolio" / "Snowball.csv"
TICKER_ALIAS_CSV = BASE / "data" / "ticker_alias.csv"
PORTFOLIO_RULES_YML = BASE / "config" / "portfolio_rules.yml"


# -----------------------
# Utilities
# -----------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_float(x) -> float | None:
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


def pct_str(x: float | None, digits: int = 1) -> str:
    if x is None:
        return ""
    return f"{x*100:.{digits}f}%"


def num_str(x: float | None, digits: int = 2) -> str:
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def read_tickers_txt(path: Path) -> list[str]:
    """
    Supports sections + comments.
    Example:
      # DK
      NOVO-B.CO
      PNDORA.CO

      # US
      V
      AAPL
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow inline comments: "AAPL  # Apple"
        s = re.split(r"\s+#", s, maxsplit=1)[0].strip()
        if s:
            tickers.append(s)
    # de-dupe keep order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def infer_country(ticker: str) -> str:
    # Robust + simple: .CO -> DK, else assume US
    t = ticker.upper()
    if t.endswith(".CO"):
        return "DK"
    return "US"


def load_index_map(path: Path) -> dict[str, str]:
    """
    index_map.csv expected: columns [Ticker, Index] OR [ticker, index]
    Returns dict {ticker -> index_string}
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        return {}
    ticker_col = cols["ticker"]
    index_col = cols.get("index")
    if not index_col:
        # allow second column fallback
        other_cols = [c for c in df.columns if c != ticker_col]
        if not other_cols:
            return {}
        index_col = other_cols[0]
    m = {}
    for _, r in df.iterrows():
        t = str(r[ticker_col]).strip()
        idx = str(r[index_col]).strip()
        if t and t.lower() != "nan":
            m[t] = "" if idx.lower() == "nan" else idx
    return m


def load_alias_map(path: Path) -> dict[str, str]:
    """
    data/ticker_alias.csv with columns: snowball,yahoo
    Maps Snowball symbol -> Yahoo symbol
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    if "snowball" not in low or "yahoo" not in low:
        return {}
    a = {}
    for _, r in df.iterrows():
        s = str(r[low["snowball"]]).strip()
        y = str(r[low["yahoo"]]).strip()
        if s and y and s.lower() != "nan" and y.lower() != "nan":
            a[s] = y
    return a


# -----------------------
# Portfolio: Snowball ingest (tolerant)
# -----------------------
def load_snowball_positions(path: Path, alias_map: dict[str, str]) -> pd.DataFrame:
    """
    Reads Snowball export and returns DataFrame with columns: Ticker, OwnedShares
    Accepts many formats by guessing column names.
    """
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "OwnedShares"])

    df = pd.read_csv(path)
    cols_l = {c.lower().strip(): c for c in df.columns}

    # Try to find symbol column
    sym_candidates = ["symbol", "ticker", "holding", "isin", "name"]
    sym_col = None
    for k in sym_candidates:
        if k in cols_l:
            sym_col = cols_l[k]
            break
    if sym_col is None:
        # fallback: first column
        sym_col = df.columns[0]

    # Try to find shares/qty column
    qty_candidates = ["shares", "quantity", "antal", "units", "ownedshares"]
    qty_col = None
    for k in qty_candidates:
        if k in cols_l:
            qty_col = cols_l[k]
            break

    if qty_col is None:
        # If no shares column, assume not usable
        return pd.DataFrame(columns=["Ticker", "OwnedShares"])

    tmp = df[[sym_col, qty_col]].copy()
    tmp.columns = ["SnowballSymbol", "OwnedShares"]
    tmp["SnowballSymbol"] = tmp["SnowballSymbol"].astype(str).str.strip()
    tmp["OwnedShares"] = pd.to_numeric(tmp["OwnedShares"], errors="coerce").fillna(0.0)

    # Map via alias if given (SnowballSymbol -> Yahoo ticker)
    tmp["Ticker"] = tmp["SnowballSymbol"].map(lambda x: alias_map.get(x, x))
    tmp["Ticker"] = tmp["Ticker"].astype(str).str.strip()

    pos = tmp.groupby("Ticker", as_index=False)["OwnedShares"].sum()
    pos = pos[pos["OwnedShares"].abs() > 1e-9].copy()
    return pos


# -----------------------
# Dividend/yield metrics (stable)
# -----------------------
def trailing_12m_dividend_and_yield(ticker: str, price: float | None) -> tuple[float | None, float | None]:
    """
    Uses yfinance dividends series. Annual = sum of last 365 days dividends.
    Yield = annual_div / price.
    """
    if price is None or price <= 0:
        return None, None
    try:
        t = yf.Ticker(ticker)
        div = t.dividends
        if div is None or len(div) == 0:
            return 0.0, 0.0
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365)
        div = div[div.index.tz_localize(None) >= cutoff]
        ann = float(div.sum()) if len(div) else 0.0
        y = ann / price if price > 0 else None
        return ann, y
    except Exception:
        return None, None


def dividend_growth_5y_cagr(ticker: str) -> float | None:
    """
    Approximates dividend growth from yearly dividend totals (calendar years).
    Uses last 6 calendar years: CAGR from year-5 to last year.
    """
    try:
        t = yf.Ticker(ticker)
        div = t.dividends
        if div is None or len(div) == 0:
            return None
        div = div.copy()
        div.index = div.index.tz_localize(None)
        # group by calendar year
        yearly = div.groupby(div.index.year).sum().sort_index()
        # Need at least 6 years to compute 5y CAGR robustly
        if len(yearly) < 6:
            return None
        last_year = int(yearly.index.max())
        start_year = last_year - 5
        if start_year not in yearly.index:
            return None
        start = float(yearly.loc[start_year])
        end = float(yearly.loc[last_year])
        if start <= 0 or end <= 0:
            return None
        cagr = (end / start) ** (1 / 5) - 1
        return cagr
    except Exception:
        return None


def fetch_quote_fields(ticker: str) -> dict:
    """
    Fetches stable fields with fallbacks.
    """
    out = {"Ticker": ticker}
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info if hasattr(tk, "fast_info") else {}

        price = safe_float(fi.get("last_price")) or safe_float(fi.get("lastPrice"))
        if price is None:
            # fallback: history
            h = tk.history(period="5d")
            if len(h):
                price = safe_float(h["Close"].iloc[-1])

        out["Price"] = price

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        out["Name"] = info.get("shortName") or info.get("longName") or ""
        out["Currency"] = info.get("currency") or fi.get("currency") or ""
        out["Sector"] = info.get("sector") or ""
        out["Exchange"] = info.get("exchange") or info.get("fullExchangeName") or ""
        out["PE"] = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))

    except Exception:
        # leave fields empty
        pass
    return out


# -----------------------
# Scoring / fair value model (simple but non-zero)
# -----------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fair_value_from_yield(annual_div: float | None, current_yield: float | None, price: float | None) -> tuple[float | None, float | None]:
    """
    Fair value using a target yield.
    If current yield is high, fair yield should usually be lower -> higher fair value.
    We use: target_yield = clamp(current_yield * 0.8, 0.015, 0.08)
    FairValue = annual_div / target_yield
    Upside% = (FairValue - price) / price
    """
    if annual_div is None or price is None or price <= 0:
        return None, None
    if current_yield is None:
        # assume 3% if unknown
        target = 0.03
    else:
        target = clamp(current_yield * 0.8, 0.015, 0.08)
    if target <= 0:
        return None, None
    fv = annual_div / target
    up = (fv - price) / price
    return fv, up


def score_row(yield_t12: float | None, divg_5y: float | None, pe: float | None) -> float:
    """
    Simple normalized score (0..100).
    """
    y = 0.0 if yield_t12 is None else clamp(yield_t12, 0, 0.12) / 0.12  # 0..1
    g = 0.0 if divg_5y is None else clamp(divg_5y, -0.10, 0.20)  # -0.1..0.2
    g = (g + 0.10) / 0.30  # 0..1
    p = 0.5
    if pe is not None and pe > 0:
        # lower PE better (within reason)
        p = 1 - clamp((pe - 8) / (30 - 8), 0, 1)  # 8->1, 30->0
    # weights
    s = 100 * (0.45 * y + 0.35 * g + 0.20 * p)
    return float(clamp(s, 0, 100))


def reco_label(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Ok"
    return "Weak"


def action_label(upside: float | None, score: float) -> str:
    if upside is None:
        return "HOLD"
    if upside >= 0.15 and score >= 70:
        return "BUY"
    if upside <= -0.10:
        return "AVOID"
    return "HOLD"


def portfolio_action(owned: bool, weight: float | None, base_action: str) -> str:
    """
    Very safe default portfolio logic:
    - If owned => HOLD (unless overweight then TRIM)
    - If not owned => follow base action
    """
    if owned:
        if weight is not None and weight >= 0.12:  # 12% cap default
            return "TRIM"
        return "HOLD"
    return base_action


# -----------------------
# HTML (DataTables + header dropdowns)
# -----------------------
def build_html(df: pd.DataFrame, updated: str) -> str:
    # DataTables expects JSON rows
    cols = list(df.columns)

    # Which columns get header dropdown filters:
    filter_cols = [
        "Country",
        "Currency",
        "Sector",
        "InIndex",
        "Owned",
        "Reco",
        "Action",
        "PortfolioAction",
    ]
    filter_idxs = [cols.index(c) for c in filter_cols if c in cols]

    data_json = df.to_dict(orient="records")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Dividend Screener (DK + US)</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
  <style>
    body {{
      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      padding: 18px;
    }}
    h1 {{ margin: 0 0 6px 0; }}
    .meta {{ color: #666; margin-bottom: 14px; }}
    table.dataTable thead th {{
      white-space: nowrap;
    }}
    thead tr.filters th {{
      padding: 6px 8px !important;
    }}
    select.dt-filter {{
      width: 100%;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>Dividend Screener (DK + US)</h1>
  <div class="meta">Updated: {updated}</div>

  <table id="tbl" class="display" style="width:100%"></table>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>

  <script>
    const columns = {json.dumps([{"title": c, "data": c} for c in cols])};
    const data = {json.dumps(data_json)};

    const filterIdxs = {json.dumps(filter_idxs)};

    $(document).ready(function() {{
      const table = $('#tbl').DataTable({{
        data: data,
        columns: columns,
        pageLength: 50,
        order: [],
        deferRender: true
      }});

      // Add a second header row for filters
      const thead = $('#tbl thead');
      const headerRow = thead.find('tr').first();
      const filterRow = $('<tr class="filters"></tr>').appendTo(thead);

      headerRow.find('th').each(function(i) {{
        const th = $('<th></th>');
        if (filterIdxs.includes(i)) {{
          const sel = $('<select class="dt-filter"><option value=""></option></select>');
          th.append(sel);

          // build unique list
          const uniq = new Set();
          table.column(i).data().each(function(v) {{
            if (v !== null && v !== undefined && String(v).trim() !== "") {{
              uniq.add(String(v));
            }}
          }});
          Array.from(uniq).sort().forEach(function(v) {{
            sel.append($('<option></option>').attr('value', v).text(v));
          }});

          sel.on('change', function() {{
            const val = $.fn.dataTable.util.escapeRegex($(this).val());
            table.column(i).search(val ? '^' + val + '$' : '', true, false).draw();
          }});
        }}
        filterRow.append(th);
      }});
    }});
  </script>
</body>
</html>
"""


# -----------------------
# Main
# -----------------------
def main():
    tickers = read_tickers_txt(TICKERS_TXT)
    index_map = load_index_map(INDEX_MAP_CSV)
    alias_map = load_alias_map(TICKER_ALIAS_CSV)

    # portfolio positions (optional)
    pos = load_snowball_positions(SNOWBALL_CSV, alias_map)
    pos_map = dict(zip(pos["Ticker"], pos["OwnedShares"])) if len(pos) else {}

    rows: list[dict] = []

    # Fetch per ticker (simple loop = stable)
    for t in tickers:
        q = fetch_quote_fields(t)
        price = safe_float(q.get("Price"))

        ann_div, yld = trailing_12m_dividend_and_yield(t, price)
        divg = dividend_growth_5y_cagr(t)

        fv, up = fair_value_from_yield(ann_div, yld, price)

        pe = safe_float(q.get("PE"))
        score = score_row(yld, divg, pe)
        reco = reco_label(score)
        act = action_label(up, score)

        country = infer_country(t)
        in_index = index_map.get(t, "")

        owned_shares = float(pos_map.get(t, 0.0)) if t in pos_map else 0.0
        owned = "Yes" if owned_shares > 0 else "No"
        owned_value = owned_shares * price if (price is not None) else 0.0

        rows.append({
            "Ticker": t,
            "Name": q.get("Name", ""),
            "Price": price,
            "DividendYield": (yld * 100) if yld is not None else None,          # numeric percent for sorting
            "DividendGrowth5Y": (divg * 100) if divg is not None else None,     # numeric percent
            "PE": pe,
            "FairValue_Yield": fv,
            "Upside_Yield_%": (up * 100) if up is not None else None,           # numeric percent
            "Score": score,
            "Reco": reco,
            "Action": act,

            "Country": country,
            "Currency": q.get("Currency", ""),
            "Sector": q.get("Sector", ""),
            "Exchange": q.get("Exchange", ""),
            "InIndex": in_index,

            "OwnedShares": owned_shares,
            "Owned": owned,
            "OwnedValue": owned_value,   # calculated later as weight too
        })

    df = pd.DataFrame(rows)

    # Portfolio weights (if any holdings)
    total_value = float(df["OwnedValue"].sum()) if "OwnedValue" in df.columns else 0.0
    if total_value > 0:
        df["Weight"] = df["OwnedValue"] / total_value
    else:
        df["Weight"] = 0.0

    # PortfolioAction
    df["PortfolioAction"] = df.apply(
        lambda r: portfolio_action(
            owned=(r.get("Owned") == "Yes"),
            weight=safe_float(r.get("Weight")),
            base_action=str(r.get("Action") or "HOLD"),
        ),
        axis=1
    )

    # Format-friendly duplicates (optional)
    # Keep numeric columns numeric for sorting in DataTables, but also add pretty strings if you want.
    # For now we keep numeric and let DataTables show raw; you can format later if desired.

    # Ensure folders
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Save HTML
    html = build_html(df, now_utc_str())
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"Wrote: {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote: {OUT_HTML}")


if __name__ == "__main__":
    main()
