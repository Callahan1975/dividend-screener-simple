#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import math
import time
import html
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance import failed. Ensure it's installed in your workflow.")
    raise

try:
    import requests  # type: ignore
except Exception:
    requests = None


# -----------------------------
# Paths
# -----------------------------
TICKERS_TXT = "tickers.txt"
MASTER_TICKERS_TXT = os.path.join("data", "tickers_master.txt")          # optional
LIVE_TICKERS_URL_TXT = os.path.join("data", "tickers_live_url.txt")      # optional (1-line URL)
ALIAS_CSV = os.path.join("data", "ticker_alias.csv")
INDEX_MAP_CSV = "index_map.csv"  # optional

PORTFOLIO_CSV_CANDIDATES = [
    os.path.join("data", "portfolio", "Snowball.csv"),
    os.path.join("data", "portfolio", "snowball.csv"),
    os.path.join("data", "portfolio", "portfolio.csv"),
    os.path.join("data", "portfolio", "portfolio_holdings.csv"),
    os.path.join("data", "portfolio.csv"),
    os.path.join("data", "portfolio_holdings.csv"),
    "portfolio.csv",
]

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


def _infer_country_from_ticker(yahoo_ticker: str) -> str:
    return "DK" if yahoo_ticker.upper().endswith(".CO") else "US"


# -----------------------------
# Dividend yield normalization (KEEP CORRECT)
# -----------------------------
def _normalize_dividend_yield(raw: Any) -> Optional[float]:
    """
    Return dividend yield as FRACTION (e.g. 0.031 = 3.1%).
    """
    v = _safe_float(raw)
    if v is None or v < 0:
        return None

    if v <= 0.30:
        return v
    if v <= 100:
        return v / 100.0
    return v / 10000.0


def _sanity_div_yield(y: Optional[float]) -> Optional[float]:
    """
    Final sanity gate. Anything above 40% yield is usually junk for normal equities.
    """
    if y is None or y < 0:
        return None
    if y > 0.40:
        return None
    return y


# -----------------------------
# Ticker parsing (robust)
# -----------------------------
def _strip_inline_comment(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return s


def _clean_ticker_line(line: str) -> str:
    s = _strip_inline_comment(line)
    if not s or s.startswith("#"):
        return ""
    s = s.split()[0].strip()
    if not s:
        return ""
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        return ""
    return s


def _clean_url_line(line: str) -> str:
    s = _strip_inline_comment(line)
    if not s or s.startswith("#"):
        return ""
    return s.strip()


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
                s = _clean_url_line(ln)
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
        r = requests.get(url, timeout=30)
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
            if not s or s in seen:
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
    print(f"[{_now_utc_str()}] Sample tickers: {merged[:15]}")
    return merged


# -----------------------------
# Alias / index map / portfolio
# -----------------------------
def _load_alias_map(alias_csv: str) -> pd.DataFrame:
    candidates = [
        alias_csv,
        os.path.join("data", "portfolio", "ticker_alias.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), "")
    if not path:
        return pd.DataFrame(columns=["input_ticker", "yahoo_ticker", "alias"])

    df = pd.read_csv(path)
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

    print(f"[{_now_utc_str()}] Alias file used: {path} (rows={len(out)})")
    return out


def _load_index_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        alt = os.path.join("src", "index_map.csv")
        if not os.path.exists(alt):
            return {}
        path = alt

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

                for key in ("yahoo_ticker", "ticker", "symbol", "Ticker", "Symbol"):
                    if key.lower() in cols:
                        c = cols[key.lower()]
                        s = set(df[c].astype(str).str.strip())
                        s = {x for x in s if x and x.lower() != "nan"}
                        print(f"[{_now_utc_str()}] Portfolio file used: {p} (tickers={len(s)})")
                        return s
                    if key in df.columns:
                        s = set(df[key].astype(str).str.strip())
                        s = {x for x in s if x and x.lower() != "nan"}
                        print(f"[{_now_utc_str()}] Portfolio file used: {p} (tickers={len(s)})")
                        return s

                print(f"[{_now_utc_str()}] Portfolio file found but no ticker column recognized: {p}")
                return set()
            except Exception as e:
                print(f"[{_now_utc_str()}] Portfolio load failed for {p}: {e}")
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
            info.get("regularMarketPrice"),
            info.get("currentPrice"),
            info.get("previousClose"),
        ))

        currency = _safe_str(_first_non_null(fast.get("currency"), info.get("currency")))
        exchange = _safe_str(_first_non_null(fast.get("exchange"), info.get("exchange")))
        sector = _safe_str(_first_non_null(info.get("sector"), info.get("industry")))
        name = _safe_str(_first_non_null(info.get("shortName"), info.get("longName"), info.get("displayName")))
        market_cap = _safe_float(info.get("marketCap"))

        raw_y = _first_non_null(info.get("dividendYield"), info.get("yield"))
        div_yield = _sanity_div_yield(_normalize_dividend_yield(raw_y))

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
# Payout warning rule (sector exceptions)
# -----------------------------
def _should_flag_payout(sector: str, payout_ratio_pct: Optional[float]) -> bool:
    """
    Only show ⚠️ / red highlight for payout > 90% in sectors where EPS-based payout is a meaningful risk signal.

    We do NOT flag by default for:
      - Real Estate (REITs)
      - Energy (midstream/utilities-like accounting)
      - Financial Services (often BDC/other)
    """
    if payout_ratio_pct is None:
        return False
    if payout_ratio_pct <= 90:
        return False

    s = (sector or "").strip().lower()

    # exceptions
    if "real estate" in s:
        return False
    if "energy" in s:
        return False
    if "financial" in s:
        return False

    return True


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


def _make_signal(total_return_pct: Optional[float], upside_pct: Optional[float], sector: str, payout_ratio_pct: Optional[float]) -> str:
    """
    Minimal "Signal" (single visual indicator):
    - GOLD: TotalReturnEst >= 15% AND Upside >= 10%
    - BUY : TotalReturnEst >= 10%
    - HOLD: TotalReturnEst 5-10%
    - WATCH: <5% OR Upside < 0
    Add ⚠️ only when payout warning rule triggers.
    """
    tr = total_return_pct
    up = upside_pct

    label = "WATCH"
    if up is not None and up < 0:
        label = "WATCH"
    elif tr is None:
        label = "WATCH"
    elif tr >= 15 and (up is not None and up >= 10):
        label = "GOLD"
    elif tr >= 10:
        label = "BUY"
    elif tr >= 5:
        label = "HOLD"
    else:
        label = "WATCH"

    warn = _should_flag_payout(sector, payout_ratio_pct)
    return f"{label}{' ⚠' if warn else ''}"


# -----------------------------
# HTML Generation (STATIC TABLE + Minimal Signal)
# -----------------------------
def _fmt_num2(v: Any) -> str:
    x = _safe_float(v)
    return "" if x is None else f"{x:,.2f}"


def _fmt_pct2(v: Any) -> str:
    x = _safe_float(v)
    return "" if x is None else f"{x:,.2f}%"


def _fmt_mcap(v: Any) -> str:
    x = _safe_float(v)
    if x is None:
        return ""
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:,.2f}T"
    if ax >= 1e9:
        return f"{x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:,.2f}M"
    if ax >= 1e3:
        return f"{x/1e3:,.2f}K"
    return f"{x:,.0f}"


def _html_escape(s: Any) -> str:
    return html.escape(_safe_str(s))


def _html_page(df: pd.DataFrame, generated_at: str) -> str:
    columns = [
        "Ticker","Alias","Name","Country","Currency","Exchange","Sector",
        "InIndex","Indexes","Portfolio","PortfolioAction","Signal","Reco","Action",
        "Price","DividendYield_%","DividendRate","PayoutRatio_%","PE","TargetMeanPrice",
        "Upside_%","Upside_Yield_%","TotalReturnEst_%","MarketCap","52W_Low","52W_High",
        "Status","Error",
    ]

    for c in columns:
        if c not in df.columns:
            df[c] = ""

    pct_cols = {"DividendYield_%","PayoutRatio_%","Upside_%","Upside_Yield_%","TotalReturnEst_%"}
    num2_cols = {"Price","DividendRate","PE","TargetMeanPrice","52W_Low","52W_High"}

    body_rows: List[str] = []
    for _, r in df[columns].iterrows():
        tds: List[str] = []
        for c in columns:
            val = r.get(c)
            if c in pct_cols:
                cell = _fmt_pct2(val)
            elif c in num2_cols:
                cell = _fmt_num2(val)
            elif c == "MarketCap":
                cell = _fmt_mcap(val)
            else:
                cell = _safe_str(val)
            tds.append(f"<td>{_html_escape(cell)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    filter_cols = ["Country","Currency","Sector","Exchange","InIndex","Signal","Reco","Action","PortfolioAction"]
    filter_html = "\n".join(
        [f"""
        <div class="filter">
          <label for="flt_{c}">{c}</label>
          <select id="flt_{c}"><option value="">All</option></select>
        </div>
        """.strip() for c in filter_cols]
    )

    cols_js = "[" + ",".join([f'"{c}"' for c in columns]) + "]"
    filters_js = "[" + ",".join([f'"{c}"' for c in filter_cols]) + "]"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
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
      display:grid; grid-template-columns:repeat(9, minmax(120px,1fr)); gap:10px; padding:12px;
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

    .badge {{
      display:inline-flex; align-items:center; gap:6px;
      padding:2px 10px; border-radius:999px;
      font-size:11px; line-height:16px;
      border:1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      white-space:nowrap;
    }}
    .dot {{ width:8px; height:8px; border-radius:999px; display:inline-block; }}
    .dot-gold {{ background: rgba(241,196,15,0.95); }}
    .dot-buy  {{ background: rgba(46,204,113,0.95); }}
    .dot-hold {{ background: rgba(200,200,200,0.9); }}
    .dot-watch{{ background: rgba(231,76,60,0.95); }}

    /* Only one extra "risk" highlight: payout warn (sector-aware) */
    .payout-bad {{ background: rgba(231,76,60,0.16) !important; }}
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
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/fixedheader/3.4.0/js/dataTables.fixedHeader.min.js"></script>

  <script>
    const COLS = {cols_js};
    const FILTERS = {filters_js};

    function escRegex(s) {{
      return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    }}

    function shouldFlagPayout(sector, payoutPct) {{
      if (!isFinite(payoutPct) || payoutPct <= 90) return false;
      const s = (sector || "").toString().toLowerCase();
      if (s.includes("real estate")) return false;
      if (s.includes("energy")) return false;
      if (s.includes("financial")) return false;
      return true;
    }}

    $(document).ready(function() {{
      const dt = $('#screener').DataTable({{
        pageLength: 50,
        order: [[ COLS.indexOf("TotalReturnEst_%"), "desc" ]],
        fixedHeader: true,
        deferRender: true,
        autoWidth: false,

        createdRow: function(row, data, dataIndex) {{
          const col = (name) => COLS.indexOf(name);
          const td = $('td', row);

          const sectorIdx = col("Sector");
          const payoutIdx = col("PayoutRatio_%");

          // payout highlight ONLY if > 90% AND sector-rule says it's meaningful
          if (payoutIdx >= 0) {{
            const sector = sectorIdx >= 0 ? (data[sectorIdx] ?? "") : "";
            const t = (data[payoutIdx] ?? "").toString().replace('%','').replace(/,/g,'').trim();
            const v = Number(t);
            if (shouldFlagPayout(sector, v)) td.eq(payoutIdx).addClass("payout-bad");
          }}

          // Signal badge
          const sigIdx = col("Signal");
          if (sigIdx >= 0) {{
            const raw = (data[sigIdx] ?? "").toString().trim();
            const lower = raw.toLowerCase();
            let dotClass = "dot-hold";
            if (lower.startsWith("gold")) dotClass = "dot-gold";
            else if (lower.startsWith("buy")) dotClass = "dot-buy";
            else if (lower.startsWith("watch")) dotClass = "dot-watch";
            else if (lower.startsWith("hold")) dotClass = "dot-hold";

            if (raw) {{
              td.eq(sigIdx).html(`<span class="badge"><span class="dot ${{dotClass}}"></span><span>${{raw}}</span></span>`);
            }}
          }}

          // Reco badge
          const recoIdx = col("Reco");
          if (recoIdx >= 0) {{
            const txt = (data[recoIdx] ?? "").toString().trim();
            if (txt) td.eq(recoIdx).html(`<span class="badge"><span>${{txt}}</span></span>`);
          }}

          // Action badge
          const actIdx = col("Action");
          if (actIdx >= 0) {{
            const txt = (data[actIdx] ?? "").toString().trim();
            if (txt) td.eq(actIdx).html(`<span class="badge"><span>${{txt}}</span></span>`);
          }}
        }}
      }});

      // Dropdown filters
      FILTERS.forEach((name) => {{
        const idx = COLS.indexOf(name);
        const sel = document.getElementById("flt_" + name);
        if (!sel || idx < 0) return;

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
          else dt.column(idx).search("^" + escRegex(val) + "$", true, false).draw();
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
                "Signal": "",
                "Reco": "", "Action": "",
                "Price": None, "DividendYield_%": None, "DividendRate": None, "PayoutRatio_%": None,
                "PE": None, "TargetMeanPrice": None,
                "Upside_%": None, "Upside_Yield_%": None, "TotalReturnEst_%": None,
                "MarketCap": None, "52W_Low": None, "52W_High": None,
                "Status": "ERROR", "Error": err or "Unknown error",
            })
            continue

        sector = _safe_str(data.get("Sector"))

        price = _safe_float(data.get("Price"))
        div_yield = _safe_float(data.get("DividendYield"))  # fraction (sanity-filtered)
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
        payout_pct = (payout * 100.0) if payout is not None else None

        signal = _make_signal(
            total_return_pct=total_return_pct,
            upside_pct=upside_pct,
            sector=sector,
            payout_ratio_pct=payout_pct,
        )

        rows.append({
            "Ticker": yahoo_ticker,
            "Alias": alias,
            "Name": _safe_str(data.get("Name")),
            "Country": country,
            "Currency": _safe_str(data.get("Currency")),
            "Exchange": _safe_str(data.get("Exchange")),
            "Sector": sector,
            "InIndex": in_index,
            "Indexes": indexes,
            "Portfolio": "Yes" if in_portfolio else "No",
            "PortfolioAction": p_action,
            "Signal": signal,
            "Reco": _safe_str(data.get("Reco")),
            "Action": action,
            "Price": _round2(price),
            "DividendYield_%": _round2(div_yield_pct),
            "DividendRate": _round2(_safe_float(data.get("DividendRate"))),
            "PayoutRatio_%": _round2(payout_pct),
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
