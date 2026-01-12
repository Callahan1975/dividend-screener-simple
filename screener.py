#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) - robust GitHub Actions runner
Outputs:
- data/screener_results.csv
- docs/index.html (DataTables + dropdown filters + nice formatting)

Fixes in this version:
- STRIPS inline comments in tickers lists (e.g. "FRC # delisted" -> "FRC")
- STRIPS anything after first whitespace token (e.g. "NA.TO (note...)" -> "NA.TO")
- Adds no-cache meta tags to HTML to reduce GitHub Pages stale caching issues
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance import failed. Ensure it's installed in your workflow.")
    raise

# Optional: requests for live tickers URL (won't break if not installed)
try:
    import requests  # type: ignore
except Exception:
    requests = None


# -----------------------------
# Paths (repo-relative)
# -----------------------------
TICKERS_TXT = "tickers.txt"
MASTER_TICKERS_TXT = os.path.join("data", "tickers_master.txt")          # optional
LIVE_TICKERS_URL_TXT = os.path.join("data", "tickers_live_url.txt")      # optional (1-line URL)
ALIAS_CSV = os.path.join("data", "ticker_alias.csv")
INDEX_MAP_CSV = "index_map.csv"  # optional
PORTFOLIO_CSV_CANDIDATES = [
    os.path.join("data", "portfolio.csv"),
    os.path.join("data", "portfolio_holdings.csv"),
    "portfolio.csv",
]  # optional

OUT_CSV = os.path.join("data", "screener_results.csv")
OUT_HTML = os.path.join("docs", "index.html")


# -----------------------------
# Helpers
# -----------------------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)) and (isinstance(x, bool) is False):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null", "na"):
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _normalize_dividend_yield(raw: Any) -> Optional[float]:
    """
    Normalize dividend yield to a fraction (e.g. 0.035 for 3.5%).
    Handles common scaling issues (e.g. 115, 700).
    """
    v = _safe_float(raw)
    if v is None or v < 0:
        return None
    if v <= 1.5:
        return v
    if v <= 100:
        return v / 100.0
    return v / 10000.0


def _infer_country_from_ticker(yahoo_ticker: str) -> str:
    t = yahoo_ticker.upper()
    if t.endswith(".CO"):
        return "DK"
    return "US"


def _first_non_null(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def _round2(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), 2)
    except Exception:
        return None


# -----------------------------
# Ticker parsing (IMPORTANT FIX)
# -----------------------------
def _clean_ticker_line(line: str) -> str:
    """
    Makes ticker lists idiot-proof:
    - removes inline comments after '#'
      e.g. "FRC # delisted" -> "FRC"
    - takes only the first whitespace-separated token
      e.g. "NA.TO (use Toronto)" -> "NA.TO"
    """
    s = (line or "").strip()
    if not s:
        return ""
    if s.startswith("#"):
        return ""

    # remove inline comment
    if "#" in s:
        s = s.split("#", 1)[0].strip()

    if not s:
        return ""

    # take only first token (kills any leftover notes)
    s = s.split()[0].strip()
    return s


def _read_lines_clean(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = _clean_ticker_line(ln)
            if t:
                out.append(t)
    return out


def _read_live_url_from_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                s = _clean_ticker_line(ln)
                if s:
                    return s
    except Exception:
        return ""
    return ""


def _fetch_tickers_from_url(url: str) -> List[str]:
    if not url:
        return []
    if requests is None:
        print("WARNING: requests not installed; cannot fetch live tickers URL. Skipping live.")
        return []
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        out: List[str] = []
        for ln in r.text.splitlines():
            t = _clean_ticker_line(ln)
            if t:
                out.append(t)
        return out
    except Exception as e:
        print(f"WARNING: Failed to fetch tickers from URL ({url}): {e}")
        return []


def _merge_unique_keep_order(lists: List[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for x in lst:
            s = (x or "").strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def _read_all_tickers() -> List[str]:
    base = _read_lines_clean(TICKERS_TXT)
    master = _read_lines_clean(MASTER_TICKERS_TXT)

    url_env = os.environ.get("TICKERS_URL", "").strip()
    url_file = _read_live_url_from_file(LIVE_TICKERS_URL_TXT)
    url = url_env or url_file

    live = _fetch_tickers_from_url(url) if url else []

    merged = _merge_unique_keep_order([base, master, live])

    src = "env" if url_env else ("file" if url_file else "none")
    print(f"[{_now_utc_str()}] Tickers sources: base={len(base)}, master={len(master)}, live={len(live)}, url_source={src} -> merged={len(merged)}")

    # Helpful debug: show a few tickers so you can verify comments are stripped
    print(f"[{_now_utc_str()}] Sample tickers: {merged[:12]}")
    return merged


# -----------------------------
# Alias / index map / portfolio
# -----------------------------
def _load_alias_map(alias_csv: str) -> pd.DataFrame:
    if not os.path.exists(alias_csv):
        return pd.DataFrame(columns=["input_ticker", "yahoo_ticker", "alias"])

    df = pd.read_csv(alias_csv)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_input = pick("input_ticker", "ticker", "symbol")
    c_yahoo = pick("yahoo_ticker", "yahoo", "yf_ticker", "yfinance", "ticker_yahoo")
    c_alias = pick("alias", "name", "display", "display_name")

    if c_input is None and c_yahoo is not None:
        c_input = c_yahoo
    if c_yahoo is None and c_input is not None:
        c_yahoo = c_input

    out = pd.DataFrame()
    out["input_ticker"] = df[c_input].astype(str).str.strip() if c_input else ""
    out["yahoo_ticker"] = df[c_yahoo].astype(str).str.strip() if c_yahoo else out["input_ticker"]
    out["alias"] = df[c_alias].astype(str).str.strip() if c_alias else ""

    out = out.fillna("")
    out["input_ticker"] = out["input_ticker"].astype(str).str.strip()
    out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
    out["alias"] = out["alias"].astype(str).str.strip()

    out = out[out["yahoo_ticker"] != ""].drop_duplicates(subset=["yahoo_ticker"], keep="first")
    return out


def _load_index_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_ticker = pick("ticker", "symbol", "yahoo_ticker", "yahoo")
    c_indexes = pick("indexes", "index", "inindex", "indices")

    if c_ticker is None or c_indexes is None:
        return {}

    mp: Dict[str, str] = {}
    for _, r in df.iterrows():
        t = _safe_str(r.get(c_ticker))
        ix = _safe_str(r.get(c_indexes))
        if t:
            mp[t] = ix
    return mp


def _load_portfolio_holdings() -> set:
    for p in PORTFOLIO_CSV_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                cols = {c.lower().strip(): c for c in df.columns}
                c_ticker = None
                for k in ("yahoo_ticker", "ticker", "symbol"):
                    if k in cols:
                        c_ticker = cols[k]
                        break
                if c_ticker is None:
                    return set()
                s = set(df[c_ticker].astype(str).str.strip())
                return {x for x in s if x}
            except Exception:
                return set()
    return set()


# -----------------------------
# yfinance fetch
# -----------------------------
def _fetch_one(ticker: str, pause_s: float = 0.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if pause_s > 0:
            time.sleep(pause_s)

        t = yf.Ticker(ticker)

        fast = {}
        try:
            fast = getattr(t, "fast_info", {}) or {}
        except Exception:
            fast = {}

        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        if not fast and not info:
            return None, "No data returned (possible delisted / quote not found)"

        price = _safe_float(_first_non_null(
            fast.get("last_price"),
            fast.get("lastPrice"),
            info.get("regularMarketPrice"),
            info.get("currentPrice"),
            info.get("previousClose"),
        ))

        currency = _safe_str(_first_non_null(fast.get("currency"), info.get("currency")))
        exchange = _safe_str(_first_non_null(fast.get("exchange"), info.get("exchange")))
        sector = _safe_str(_first_non_null(info.get("sector"), info.get("industry")))
        name = _safe_str(_first_non_null(info.get("shortName"), info.get("longName"), info.get("displayName")))
        market_cap = _safe_float(info.get("marketCap"))

        div_yield = _normalize_dividend_yield(_first_non_null(info.get("dividendYield"), info.get("yield")))
        div_rate = _safe_float(info.get("dividendRate"))
        payout_ratio = _safe_float(info.get("payoutRatio"))
        pe = _safe_float(_first_non_null(info.get("trailingPE"), info.get("forwardPE")))

        target_mean = _safe_float(_first_non_null(info.get("targetMeanPrice"), info.get("targetMedianPrice")))
        reco = _safe_str(_first_non_null(info.get("recommendationKey"), info.get("recommendationMean")))

        low_52 = _safe_float(_first_non_null(fast.get("year_low"), info.get("fiftyTwoWeekLow")))
        high_52 = _safe_float(_first_non_null(fast.get("year_high"), info.get("fiftyTwoWeekHigh")))

        return {
            "Name": name,
            "Price": price,
            "Currency": currency,
            "Exchange": exchange,
            "Sector": sector,
            "MarketCap": market_cap,
            "DividendYield": div_yield,  # fraction
            "DividendRate": div_rate,
            "PayoutRatio": payout_ratio,
            "PE": pe,
            "TargetMeanPrice": target_mean,
            "Reco": reco,
            "52W_Low": low_52,
            "52W_High": high_52,
        }, None

    except Exception as e:
        return None, str(e)


# -----------------------------
# Scoring / actions
# -----------------------------
def _calc_upside(price: Optional[float], target: Optional[float]) -> Optional[float]:
    if price is None or target is None or price <= 0:
        return None
    return (target / price) - 1.0


def _calc_total_return_est(div_yield: Optional[float], upside: Optional[float]) -> Optional[float]:
    if div_yield is None and upside is None:
        return None
    return (div_yield or 0.0) + (upside or 0.0)


def _make_action(total_return_est: Optional[float], div_yield: Optional[float]) -> str:
    if total_return_est is None:
        if div_yield is not None:
            if div_yield >= 0.04:
                return "HOLD"
            if div_yield >= 0.02:
                return "WATCH"
        return "WATCH"

    if total_return_est >= 0.15:
        return "STRONG_BUY"
    if total_return_est >= 0.10:
        return "BUY"
    if total_return_est >= 0.05:
        return "HOLD"
    return "WATCH"


def _portfolio_action(is_in_portfolio: bool, action: str) -> str:
    if is_in_portfolio:
        return "REVIEW" if action == "WATCH" else "HOLD"
    return "NEW_CANDIDATE" if action in ("BUY", "STRONG_BUY") else ""


# -----------------------------
# HTML Generation
# -----------------------------
def _html_page(df: pd.DataFrame, generated_at: str) -> str:
    columns = [
        "Ticker","Alias","Name","Country","Currency","Exchange","Sector",
        "InIndex","Indexes","Portfolio","PortfolioAction","Reco","Action",
        "Price","DividendYield_%","DividendRate","PayoutRatio_%","PE","TargetMeanPrice",
        "Upside_%","Upside_Yield_%","TotalReturnEst_%","MarketCap","52W_Low","52W_High",
        "Status","Error",
    ]

    for c in columns:
        if c not in df.columns:
            df[c] = ""

    records = df[columns].to_dict(orient="records")
    data_json = json.dumps(records, ensure_ascii=False)

    filter_cols = ["Country","Currency","Sector","Exchange","InIndex","Reco","Action","PortfolioAction"]
    filter_html = "\n".join(
        [f"""
        <div class="filter">
          <label for="flt_{c}">{c}</label>
          <select id="flt_{c}"><option value="">All</option></select>
        </div>
        """.strip() for c in filter_cols]
    )

    pct_cols = {"DividendYield_%","PayoutRatio_%","Upside_%","Upside_Yield_%","TotalReturnEst_%"}
    num_2_cols = {"Price","DividendRate","PE","TargetMeanPrice","52W_Low","52W_High"}
    market_cap_col = "MarketCap"

    col_defs = []
    for idx, c in enumerate(columns):
        if c in pct_cols:
            col_defs.append({"targets": idx, "render": "function(data,type,row){return renderPercent(data,type);}"} )
        elif c == market_cap_col:
            col_defs.append({"targets": idx, "render": "function(data,type,row){return renderMarketCap(data,type);}"} )
        elif c in num_2_cols:
            col_defs.append({"targets": idx, "render": "function(data,type,row){return renderNumber2(data,type);}"} )

    col_defs_js = json.dumps(col_defs, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>

  <!-- Reduce stale caching on GitHub Pages -->
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
  <meta http-equiv="Pragma" content="no-cache"/>
  <meta http-equiv="Expires" content="0"/>

  <title>Dividend Screener (DK + US)</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.4.0/css/fixedHeader.dataTables.min.css"/>

  <style>
    :root {{
      --bg:#0b0f19; --panel:#121a2a; --text:#e8eefc; --muted:#a9b6d3; --border:rgba(255,255,255,0.12);
    }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:20px 18px 12px; border-bottom:1px solid var(--border); background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(0,0,0,0)); }}
    header h1 {{ margin:0; font-size:18px; }}
    header .meta {{ margin-top:6px; font-size:12px; color:var(--muted); }}
    .wrap {{ padding:14px 14px 28px; }}
    .filters {{
      display:grid; grid-template-columns:repeat(8, minmax(120px,1fr)); gap:10px; padding:12px;
      background:var(--panel); border:1px solid var(--border); border-radius:14px; margin-bottom:12px;
    }}
    .filter label {{ display:block; font-size:11px; color:var(--muted); margin-bottom:6px; }}
    .filter select {{
      width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--border);
      background:rgba(255,255,255,0.04); color:var(--text); outline:none;
    }}
    .tableWrap {{ padding:12px; background:var(--panel); border:1px solid var(--border); border-radius:14px; }}
    table.dataTable {{ width:100% !important; color:var(--text); background:transparent; }}
    table.dataTable thead th {{ color:var(--muted); border-bottom:1px solid var(--border) !important; }}
    table.dataTable tbody td {{ border-top:1px solid rgba(255,255,255,0.06) !important; }}
    .dataTables_wrapper .dataTables_filter input,
    .dataTables_wrapper .dataTables_length select {{
      background:rgba(255,255,255,0.04) !important; color:var(--text) !important; border:1px solid var(--border) !important;
      border-radius:10px !important; padding:6px 10px !important; outline:none;
    }}
  </style>
</head>

<body>
  <header>
    <h1>Dividend Screener (DK + US)</h1>
    <div class="meta">Generated: {generated_at} · Rows: {len(df)}</div>
  </header>

  <div class="wrap">
    <div class="filters">{filter_html}</div>
    <div class="tableWrap">
      <table id="screener" class="display" style="width:100%">
        <thead><tr>{''.join([f'<th>{c}</th>' for c in columns])}</tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/fixedheader/3.4.0/js/dataTables.fixedHeader.min.js"></script>

  <script>
    const DATA = {data_json};

    function asNumber(x) {{
      if (x === null || x === undefined) return null;
      if (typeof x === "number") return isFinite(x) ? x : null;
      const s = String(x).trim();
      if (!s) return null;
      const v = Number(s);
      return isFinite(v) ? v : null;
    }}

    const nf2 = new Intl.NumberFormat(undefined, {{ minimumFractionDigits:2, maximumFractionDigits:2 }});
    const nf0 = new Intl.NumberFormat(undefined, {{ maximumFractionDigits:0 }});

    function renderNumber2(data, type) {{
      const v = asNumber(data);
      if (v === null) return (type === "sort") ? -Infinity : "";
      return (type === "sort") ? v : nf2.format(v);
    }}
    function renderPercent(data, type) {{
      const v = asNumber(data);
      if (v === null) return (type === "sort") ? -Infinity : "";
      return (type === "sort") ? v : (nf2.format(v) + "%");
    }}
    function renderMarketCap(data, type) {{
      const v = asNumber(data);
      if (v === null) return (type === "sort") ? -Infinity : "";
      if (type === "sort") return v;
      const abs = Math.abs(v);
      if (abs >= 1e12) return nf2.format(v/1e12) + "T";
      if (abs >= 1e9)  return nf2.format(v/1e9)  + "B";
      if (abs >= 1e6)  return nf2.format(v/1e6)  + "M";
      if (abs >= 1e3)  return nf2.format(v/1e3)  + "K";
      return nf0.format(v);
    }}

    $(document).ready(function() {{
      const columns = {json.dumps(columns, ensure_ascii=False)}.map(c => ({{ data: c }}));
      const dt = $('#screener').DataTable({{
        data: DATA,
        columns: columns,
        pageLength: 50,
        order: [[ columns.findIndex(x => x.data === "TotalReturnEst_%"), "desc" ]],
        fixedHeader: true,
        deferRender: true,
        autoWidth: false,
        columnDefs: {col_defs_js},
      }});

      const filterCols = {json.dumps(filter_cols, ensure_ascii=False)};
      filterCols.forEach(colName => {{
        const idx = columns.findIndex(x => x.data === colName);
        if (idx < 0) return;
        const sel = document.getElementById("flt_" + colName);
        if (!sel) return;

        const values = new Set();
        dt.column(idx).data().each(function(v) {{
          const s = (v ?? "").toString().trim();
          if (s) values.add(s);
        }});
        Array.from(values).sort((a,b)=>a.localeCompare(b)).forEach(v => {{
          const opt = document.createElement("option");
          opt.value = v; opt.textContent = v;
          sel.appendChild(opt);
        }});

        sel.addEventListener("change", () => {{
          const val = sel.value;
          if (!val) dt.column(idx).search("").draw();
          else dt.column(idx).search("^" + val.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + "$", true, false).draw();
        }});
      }});
    }});
  </script>
</body>
</html>
"""


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    print(f"[{_now_utc_str()}] Starting screener.py")

    raw_tickers = _read_all_tickers()
    if not raw_tickers:
        print(f"ERROR: No tickers found. Ensure {TICKERS_TXT} exists and has tickers.")
        return 2

    alias_df = _load_alias_map(ALIAS_CSV)
    alias_by_input: Dict[str, str] = {}
    yahoo_by_input: Dict[str, str] = {}

    for _, r in alias_df.iterrows():
        inp = _safe_str(r.get("input_ticker"))
        yah = _safe_str(r.get("yahoo_ticker"))
        ali = _safe_str(r.get("alias"))
        if inp:
            yahoo_by_input[inp] = yah or inp
            alias_by_input[inp] = ali

    tickers: List[Tuple[str, str]] = []
    for t in raw_tickers:
        yahoo_t = yahoo_by_input.get(t, t)
        tickers.append((t, yahoo_t))

    index_map = _load_index_map(INDEX_MAP_CSV)
    portfolio_set = _load_portfolio_holdings()

    rows: List[Dict[str, Any]] = []
    failures = 0

    for input_ticker, yahoo_ticker in tickers:
        alias = alias_by_input.get(input_ticker, "")
        country = _infer_country_from_ticker(yahoo_ticker)

        data, err = _fetch_one(yahoo_ticker, pause_s=0.0)

        if data is None:
            failures += 1
            rows.append({
                "Ticker": yahoo_ticker, "Alias": alias, "Name": "",
                "Country": country, "Currency": "", "Exchange": "", "Sector": "",
                "InIndex": "", "Indexes": "",
                "Portfolio": "Yes" if yahoo_ticker in portfolio_set else "No",
                "PortfolioAction": "",
                "Reco": "", "Action": "",
                "Price": None, "DividendYield_%": None, "DividendRate": None, "PayoutRatio_%": None,
                "PE": None, "TargetMeanPrice": None,
                "Upside_%": None, "Upside_Yield_%": None, "TotalReturnEst_%": None,
                "MarketCap": None, "52W_Low": None, "52W_High": None,
                "Status": "ERROR", "Error": err or "Unknown error",
            })
            print(f"  - FAIL {yahoo_ticker}: {err}")
            continue

        price = _safe_float(data.get("Price"))
        div_yield = _safe_float(data.get("DividendYield"))  # fraction
        payout = _safe_float(data.get("PayoutRatio"))
        target = _safe_float(data.get("TargetMeanPrice"))

        upside = _calc_upside(price, target)
        total_return_est = _calc_total_return_est(div_yield, upside)

        action = _make_action(total_return_est, div_yield)
        in_portfolio = yahoo_ticker in portfolio_set
        p_action = _portfolio_action(in_portfolio, action)

        indexes = index_map.get(yahoo_ticker, "")
        in_index = "Yes" if indexes.strip() else "No"

        div_yield_pct = (div_yield * 100.0) if div_yield is not None else None
        upside_pct = (upside * 100.0) if upside is not None else None
        upside_yield_pct = None
        if div_yield_pct is not None or upside_pct is not None:
            upside_yield_pct = (div_yield_pct or 0.0) + (upside_pct or 0.0)
        total_return_pct = (total_return_est * 100.0) if total_return_est is not None else None

        rows.append({
            "Ticker": yahoo_ticker,
            "Alias": alias,
            "Name": _safe_str(data.get("Name")),
            "Country": country,
            "Currency": _safe_str(data.get("Currency")),
            "Exchange": _safe_str(data.get("Exchange")),
            "Sector": _safe_str(data.get("Sector")),
            "InIndex": in_index,
            "Indexes": indexes,
            "Portfolio": "Yes" if in_portfolio else "No",
            "PortfolioAction": p_action,
            "Reco": _safe_str(data.get("Reco")),
            "Action": action,
            "Price": _round2(price),
            "DividendYield_%": _round2(div_yield_pct),
            "DividendRate": _round2(_safe_float(data.get("DividendRate"))),
            "PayoutRatio_%": _round2((payout * 100.0) if payout is not None else None),
            "PE": _round2(_safe_float(data.get("PE"))),
            "TargetMeanPrice": _round2(target),
            "Upside_%": _round2(upside_pct),
            "Upside_Yield_%": _round2(upside_yield_pct),
            "TotalReturnEst_%": _round2(total_return_pct),
            "MarketCap": _safe_float(data.get("MarketCap")),
            "52W_Low": _round2(_safe_float(data.get("52W_Low"))),
            "52W_High": _round2(_safe_float(data.get("52W_High"))),
            "Status": "OK",
            "Error": "",
        })

    df = pd.DataFrame(rows)

    df["TotalReturnEst_%_sort"] = pd.to_numeric(df.get("TotalReturnEst_%"), errors="coerce")
    df["DividendYield_%_sort"] = pd.to_numeric(df.get("DividendYield_%"), errors="coerce")
    df = df.sort_values(
        by=["Status", "TotalReturnEst_%_sort", "DividendYield_%_sort"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["TotalReturnEst_%_sort", "DividendYield_%_sort"], errors="ignore")

    _ensure_dir(OUT_CSV)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[{_now_utc_str()}] Wrote CSV: {OUT_CSV} ({len(df)} rows, {failures} failures)")

    _ensure_dir(OUT_HTML)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(_html_page(df, generated_at=_now_utc_str()))
    print(f"[{_now_utc_str()}] Wrote HTML: {OUT_HTML}")

    ok_rows = int((df["Status"] == "OK").sum()) if "Status" in df.columns else 0
    if ok_rows == 0:
        print("ERROR: No successful ticker fetches. Failing job.")
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("FATAL: Unhandled error in screener.py")
        print(str(e))
        traceback.print_exc()
        sys.exit(10)
