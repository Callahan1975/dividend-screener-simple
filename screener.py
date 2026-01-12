#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) — Stable UI + Correct % formatting

Outputs:
- data/screener_results.csv
- docs/index.html   (GitHub Pages friendly)
"""

from __future__ import annotations

import math
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Add it to requirements.txt") from e


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

TICKERS_TXT = ROOT / "tickers.txt"
ALIAS_CSV = DATA_DIR / "ticker_alias.csv"

OUT_CSV = DATA_DIR / "screener_results.csv"
OUT_HTML = DOCS_DIR / "index.html"


# ----------------------------
# Helpers
# ----------------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        v = float(str(x).replace(",", "").strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Create it with one ticker per line.")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # allow "TICKER  # comment"
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def load_alias_map(path: Path) -> dict[str, str]:
    """
    CSV format:
    snowball,yahoo
    NOVO-B,NOVO-B.CO
    ...
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    if "snowball" not in df.columns or "yahoo" not in df.columns:
        return {}

    m: dict[str, str] = {}
    for _, r in df.iterrows():
        a = str(r["snowball"]).strip()
        y = str(r["yahoo"]).strip()
        if a and y and a.lower() != "nan" and y.lower() != "nan":
            m[a] = y
    return m


def map_to_yahoo(ticker: str, alias: dict[str, str]) -> str:
    t = ticker.strip()
    return alias.get(t, t)


def pct(x: float | None) -> float | None:
    """Return percent value (0-100) for an input ratio (0-1)."""
    if x is None:
        return None
    return x * 100.0


def compute_dividend_yield(info: dict, price: float | None) -> float | None:
    """
    Correct yield (ratio, not percent): e.g. 0.0721 for 7.21%
    yfinance has:
      - trailingAnnualDividendRate (currency)
      - trailingAnnualDividendYield (ratio)
    We prefer rate/price to avoid weird scaling.
    """
    if price is None or price <= 0:
        return None

    rate = safe_float(info.get("trailingAnnualDividendRate"))
    yld = safe_float(info.get("trailingAnnualDividendYield"))

    # Most stable: rate / price
    if rate is not None and rate >= 0:
        val = rate / price
        if 0 <= val <= 1.5:  # allow high but sane
            return val

    # fallback
    if yld is not None and 0 <= yld <= 1.5:
        return yld

    return None


def compute_div_growth_5y(div_series: pd.Series) -> float | None:
    """
    dividend series: cash dividends indexed by date (from yfinance actions["Dividends"])
    Use yearly totals and CAGR over ~5 years if possible.
    Returns ratio (0.10 = 10%).
    """
    if div_series is None or len(div_series) == 0:
        return None

    s = div_series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return None

    yearly = s.resample("Y").sum()
    yearly = yearly[yearly > 0]
    if len(yearly) < 6:
        return None

    # last 6 year points gives 5-year span between first and last
    y0 = safe_float(yearly.iloc[-6])
    y1 = safe_float(yearly.iloc[-1])
    if not y0 or not y1 or y0 <= 0 or y1 <= 0:
        return None

    cagr = (y1 / y0) ** (1 / 5) - 1
    if math.isnan(cagr) or math.isinf(cagr):
        return None
    # clamp to avoid insane values from special dividends
    return clamp(cagr, -0.5, 1.5)


def is_dk_yahoo(yahoo_ticker: str) -> bool:
    return yahoo_ticker.upper().endswith(".CO")


def yield_for_model(current_yield: float | None, dk: bool) -> float | None:
    """
    This is NOT the current yield.
    It's a "normalised required yield" used for fair value by yield.

    If yield is crazy because of special dividends, we cap hard.
    """
    if current_yield is None:
        return None

    # Normal ranges:
    # DK large caps often 1-4%, US 1-6% typical for DGI.
    hi = 0.08 if dk else 0.10
    lo = 0.015 if dk else 0.015

    y = clamp(current_yield, lo, hi)

    # push towards a "required yield" not equal to current yield (to avoid always 0% upside)
    # simple heuristic: required yield a bit lower than current for quality names
    req = clamp(y * 0.85, lo, hi)
    return req


def fair_value_by_yield(price: float | None, annual_div: float | None, req_yield: float | None) -> float | None:
    if price is None or annual_div is None or req_yield is None:
        return None
    if req_yield <= 0:
        return None
    fv = annual_div / req_yield
    if fv <= 0 or math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def upside_pct(price: float | None, fair_value: float | None) -> float | None:
    if price is None or fair_value is None or price <= 0:
        return None
    return (fair_value / price - 1.0)


def score_row(req_yield: float | None, div_growth_5y: float | None, pe: float | None) -> float:
    """
    Simple scoring: higher growth, reasonable PE, and yield not too low.
    Outputs 0..100
    """
    base = 50.0

    if div_growth_5y is not None:
        base += clamp(div_growth_5y, -0.2, 0.3) * 100  # up to +30

    if pe is not None and pe > 0:
        if pe < 10:
            base += 10
        elif pe < 18:
            base += 5
        elif pe > 35:
            base -= 10

    if req_yield is not None:
        base += clamp(req_yield, 0.015, 0.06) * 200  # up to +12

    return float(clamp(base, 0, 100))


def reco_label(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Ok"
    return "Weak"


def action_label(upside: float | None, score: float) -> str:
    if upside is None:
        return "HOLD"
    if upside >= 0.20 and score >= 70:
        return "BUY"
    if upside <= -0.15:
        return "AVOID"
    return "HOLD"


# ----------------------------
# Fetch
# ----------------------------
def fetch_one(yahoo_ticker: str) -> dict:
    """
    Returns row dict. Never raises on missing ticker; logs and returns minimal.
    """
    row = {
        "Ticker": yahoo_ticker,
        "Name": None,
        "Price": None,
        "DividendYield": None,          # ratio
        "DividendGrowth5Y": None,       # ratio
        "PayoutRatio": None,            # ratio
        "PE": None,
        "FairValue_Yield": None,
        "Upside_Yield_%": None,         # ratio
        "Score": None,
        "Reco": None,
        "Action": None,
        "Country": None,
        "Currency": None,
        "Sector": None,
        "Exchange": None,
        "InIndex": 0,
    }

    t = yf.Ticker(yahoo_ticker)

    # price
    price = None
    try:
        fi = t.fast_info
        price = safe_float(getattr(fi, "last_price", None) or fi.get("last_price"))
    except Exception:
        price = None

    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    if price is None:
        # fallback to info
        price = safe_float(info.get("currentPrice")) or safe_float(info.get("regularMarketPrice"))

    row["Price"] = price
    row["Name"] = info.get("shortName") or info.get("longName") or info.get("displayName")

    # meta
    row["Currency"] = info.get("currency")
    row["Exchange"] = info.get("exchange")
    row["Sector"] = info.get("sector")
    row["Country"] = "DK" if is_dk_yahoo(yahoo_ticker) else (info.get("country") or "US")
    if row["Country"] not in ("DK", "US"):
        # normalize
        row["Country"] = "DK" if is_dk_yahoo(yahoo_ticker) else "US"

    # dividend yield (ratio)
    yld = compute_dividend_yield(info, price)
    row["DividendYield"] = yld

    # annual dividend from trailing rate if possible
    annual_div = safe_float(info.get("trailingAnnualDividendRate"))
    if annual_div is None and (yld is not None and price is not None):
        annual_div = yld * price

    # payout ratio (ratio)
    pr = safe_float(info.get("payoutRatio"))
    if pr is not None and pr > 5:
        # Sometimes comes as percent; fix if needed
        pr = pr / 100.0
    row["PayoutRatio"] = pr if pr is None else clamp(pr, 0, 2.0)

    # PE
    pe = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))
    row["PE"] = pe

    # div growth 5y
    dg = None
    try:
        actions = t.actions
        if actions is not None and "Dividends" in actions.columns:
            dg = compute_div_growth_5y(actions["Dividends"])
    except Exception:
        dg = None
    row["DividendGrowth5Y"] = dg

    # fair value by yield
    dk = is_dk_yahoo(yahoo_ticker)
    req = yield_for_model(yld, dk)
    fv = fair_value_by_yield(price, annual_div, req)
    up = upside_pct(price, fv)

    row["FairValue_Yield"] = fv
    row["Upside_Yield_%"] = up

    sc = score_row(req, dg, pe)
    row["Score"] = sc
    row["Reco"] = reco_label(sc)
    row["Action"] = action_label(up, sc)

    return row


def load_index_map() -> dict[str, int]:
    """
    Optional: index_map.csv exists in repo root.
    Expected columns: Ticker,InIndex (0/1) or similar.
    If missing, returns {}.
    """
    path = ROOT / "index_map.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # try common patterns
    tcol = "ticker" if "ticker" in cols else (cols[0] if cols else None)
    icol = "inindex" if "inindex" in cols else None
    if not tcol:
        return {}

    out = {}
    for _, r in df.iterrows():
        t = str(r.get(tcol, "")).strip()
        if not t:
            continue
        v = r.get(icol, 0) if icol else r.get("index", 0)
        out[t.upper()] = int(safe_float(v) or 0)
    return out


# ----------------------------
# HTML (DataTables + dropdown filters)
# ----------------------------
def fmt_num(x, decimals=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return ""


def fmt_pct_ratio(x, decimals=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x)*100:.{decimals}f}%"
    except Exception:
        return ""


def build_html(df: pd.DataFrame, updated_str: str) -> str:
    # Columns + how we want to display them
    cols = list(df.columns)

    # Which columns should be dropdown filters:
    dropdown_cols = ["Reco", "Action", "Country", "Currency", "Sector", "Exchange", "InIndex", "PortfolioAction"]

    # Add PortfolioAction etc if absent
    for c in ["OwnedShares", "OwnedValue", "Weight", "PortfolioAction"]:
        if c not in cols:
            df[c] = 0 if c != "PortfolioAction" else ""

    # Prepare render formatting in JS via columnDefs
    # We'll format numeric + percent columns in JS for consistent view.
    percent_cols = {"DividendYield", "DividendGrowth5Y", "Upside_Yield_%", "Weight", "PayoutRatio"}
    money_cols = {"Price", "FairValue_Yield", "OwnedValue"}

    # Column indices
    col_idx = {c: i for i, c in enumerate(df.columns)}
    percent_idx = [col_idx[c] for c in percent_cols if c in col_idx]
    money_idx = [col_idx[c] for c in money_cols if c in col_idx]
    score_idx = col_idx.get("Score", None)

    data_records = df.to_dict(orient="records")

    # Escape-safe JSON
    data_json = json.dumps(data_records, ensure_ascii=False)

    dropdown_idx = [col_idx[c] for c in dropdown_cols if c in col_idx]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dividend Screener (DK + US)</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 20px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .meta {{ color: #666; margin: 0 0 16px 0; }}
    .filters {{ display:flex; flex-wrap:wrap; gap:10px; margin: 8px 0 12px 0; }}
    .filters label {{ font-size: 12px; color:#444; display:flex; flex-direction:column; gap:4px; }}
    .filters select {{ min-width: 160px; padding: 6px; }}
    table.dataTable thead th {{ white-space: nowrap; }}
    td {{ white-space: nowrap; }}
  </style>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
</head>
<body>
  <h1>Dividend Screener (DK + US)</h1>
  <p class="meta">Updated: {updated_str}</p>

  <div class="filters" id="filters"></div>

  <table id="tbl" class="display" style="width:100%"></table>

<script>
const data = {data_json};
const columns = {json.dumps([{"title": c, "data": c} for c in df.columns], ensure_ascii=False)};
const dropdownCols = {json.dumps(dropdown_idx)};

function uniqSorted(values) {{
  const set = new Set(values.filter(v => v !== null && v !== undefined && String(v).trim() !== ""));
  return Array.from(set).map(v => String(v)).sort();
}}

$(document).ready(function() {{
  const table = $('#tbl').DataTable({{
    data: data,
    columns: columns,
    pageLength: 50,
    order: [[{score_idx if score_idx is not None else 0}, 'desc']],
    columnDefs: [
      {{
        targets: {money_idx},
        render: function(data, type, row) {{
          const n = parseFloat(data);
          if (isNaN(n)) return '';
          return n.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
        }}
      }},
      {{
        targets: {percent_idx},
        render: function(data, type, row) {{
          const n = parseFloat(data);
          if (isNaN(n)) return '';
          // data is ratio (0.07) -> display 7.00%
          return (n*100).toFixed(2) + '%';
        }}
      }},
      {{
        targets: [{score_idx if score_idx is not None else -1}],
        render: function(data, type, row) {{
          const n = parseFloat(data);
          if (isNaN(n)) return '';
          return n.toFixed(1);
        }}
      }}
    ]
  }});

  // Build dropdown filters
  const filtersDiv = document.getElementById('filters');

  dropdownCols.forEach(idx => {{
    const col = table.column(idx);
    const title = columns[idx].title;

    const label = document.createElement('label');
    label.textContent = title;

    const sel = document.createElement('select');
    const optAll = document.createElement('option');
    optAll.value = '';
    optAll.textContent = 'All';
    sel.appendChild(optAll);

    const values = [];
    col.data().each(function(v) {{ values.push(v); }});
    uniqSorted(values).forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    }});

    sel.addEventListener('change', function() {{
      const val = this.value;
      if (!val) {{
        col.search('').draw();
      }} else {{
        // exact match
        col.search('^' + $.fn.dataTable.util.escapeRegex(val) + '$', true, false).draw();
      }}
    }});

    label.appendChild(sel);
    filtersDiv.appendChild(label);
  }});
}});
</script>

</body>
</html>
"""
    return html


# ----------------------------
# Main
# ----------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    alias = load_alias_map(ALIAS_CSV)
    tickers = read_tickers(TICKERS_TXT)
    yahoo = [map_to_yahoo(t, alias) for t in tickers]

    index_map = load_index_map()

    rows = []
    for i, yt in enumerate(yahoo, 1):
        try:
            r = fetch_one(yt)
            # index flag
            r["InIndex"] = int(index_map.get(str(r["Ticker"]).upper(), 0))
            rows.append(r)
        except Exception as e:
            # never crash whole run
            print(f"ERROR fetching {yt}: {e}")
        time.sleep(0.1)

    df = pd.DataFrame(rows)

    # Ensure numeric columns are numeric
    numeric_cols = [
        "Price", "DividendYield", "DividendGrowth5Y", "PayoutRatio", "PE",
        "FairValue_Yield", "Upside_Yield_%", "Score", "OwnedShares", "OwnedValue", "Weight"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create readable “percent-string” columns? (CSV stays numeric ratios; HTML formats nicely)
    # Sorting: best first
    df = df.sort_values(["Score", "Upside_Yield_%"], ascending=[False, False], na_position="last")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Save HTML
    html = build_html(df, now_utc_str())
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
