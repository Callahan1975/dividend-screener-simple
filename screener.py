#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) - GitHub Actions friendly

INPUT (choose one):
  1) data/tickers.csv with a column named: Ticker
  2) data/tickers.txt with one ticker per line
Optional:
  - data/index_map.csv with columns:
      Ticker, Indexes
    where Indexes can be "S&P 500|Nasdaq 100|OMXC25" etc (pipe-separated)

OUTPUT:
  - data/screener_results.csv
  - docs/index.html

Notes:
  - Works for DK tickers (e.g. NOVO-B.CO) and US tickers (e.g. V, KO).
  - Skips invalid tickers like ".CO" and blanks.
  - Robust to missing Yahoo fields (won't crash the run).
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Add it to your workflow pip install.")
    raise


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")

TICKERS_CSV = DATA_DIR / "tickers.csv"
TICKERS_TXT = DATA_DIR / "tickers.txt"
INDEX_MAP_CSV = DATA_DIR / "index_map.csv"

OUT_CSV = DATA_DIR / "screener_results.csv"
OUT_HTML = DOCS_DIR / "index.html"

# Fetch tuning
BATCH_SIZE = 40
SLEEP_BETWEEN_BATCHES_SEC = 1.0

# Basic scoring weights (tweak as you like)
WEIGHT_YIELD = 0.35
WEIGHT_GROWTH = 0.35
WEIGHT_VALUATION = 0.30


# -------------------------
# HELPERS
# -------------------------

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_ticker(t: str) -> bool:
    if t is None:
        return False
    t = str(t).strip()
    if not t:
        return False
    if t.startswith("."):
        return False
    # prevent ".CO" alone or weird fragments
    if t.upper() in {".CO", ".TO", ".ST"}:
        return False
    if t.endswith(".CO") and len(t) <= 3:
        return False
    return True


def read_tickers() -> list[str]:
    tickers: list[str] = []

    if TICKERS_CSV.exists():
        df = pd.read_csv(TICKERS_CSV)
        # accept "Ticker" or first column
        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
        tickers = [str(x).strip() for x in df[col].tolist()]
    elif TICKERS_TXT.exists():
        tickers = [line.strip() for line in TICKERS_TXT.read_text(encoding="utf-8").splitlines()]
    else:
        raise FileNotFoundError(
            "No tickers input found. Create either data/tickers.csv (Ticker column) "
            "or data/tickers.txt (one per line)."
        )

    tickers = [t for t in tickers if is_valid_ticker(t)]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def read_index_map() -> dict[str, str]:
    """
    Returns dict: ticker -> "Index1 | Index2"
    """
    if not INDEX_MAP_CSV.exists():
        return {}
    df = pd.read_csv(INDEX_MAP_CSV)
    if "Ticker" not in df.columns:
        # tolerate lowercase
        df.columns = [c.strip() for c in df.columns]
    if "Ticker" not in df.columns:
        return {}

    if "Indexes" not in df.columns:
        # attempt alternative names
        for alt in ["Index", "IndexName", "IndexList"]:
            if alt in df.columns:
                df["Indexes"] = df[alt]
                break
    if "Indexes" not in df.columns:
        return {}

    out = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip()
        if not is_valid_ticker(t):
            continue
        idx = "" if pd.isna(r["Indexes"]) else str(r["Indexes"]).strip()
        # normalize separators
        idx = idx.replace(";", "|").replace(",", "|")
        # make pretty
        idx = " | ".join([x.strip() for x in idx.split("|") if x.strip()])
        out[t] = idx
    return out


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if pd.isna(x):
                return None
            return float(x)
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def pct(a: float | None) -> float | None:
    if a is None:
        return None
    return a * 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_row(div_yield: float | None, div_growth_5y: float | None, pe: float | None) -> float:
    """
    Simple composite score: yield + growth + valuation.
    All sub-scores are normalized to 0..100 and weighted.
    """
    # Yield score: 0% -> 0, 6% -> 100 (cap)
    if div_yield is None:
        s_y = 0.0
    else:
        s_y = clamp((div_yield / 0.06) * 100.0, 0.0, 100.0)

    # Growth score: 0% -> 0, 15% -> 100 (cap)
    if div_growth_5y is None:
        s_g = 0.0
    else:
        s_g = clamp((div_growth_5y / 0.15) * 100.0, 0.0, 100.0)

    # Valuation score: PE 10 -> 100, PE 30 -> 0 (linear)
    if pe is None or pe <= 0:
        s_v = 40.0  # neutral default if missing
    else:
        s_v = clamp((30.0 - pe) / (30.0 - 10.0) * 100.0, 0.0, 100.0)

    score = WEIGHT_YIELD * s_y + WEIGHT_GROWTH * s_g + WEIGHT_VALUATION * s_v
    return float(round(score, 2))


def recommendation(score: float, div_yield: float | None, pe: float | None) -> str:
    """
    Simple rule-based recommendation label.
    """
    # You can tune thresholds here.
    if score >= 75 and (div_yield is None or div_yield >= 0.015):
        return "BUY"
    if score >= 55:
        return "HOLD"
    return "WATCH"


def pill_class(label: str) -> str:
    label = (label or "").upper()
    if label == "BUY":
        return "pill-green"
    if label == "HOLD":
        return "pill-amber"
    if label == "WATCH":
        return "pill-gray"
    return "pill-gray"


def format_num(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return ""


def format_pct(x, digits=1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return ""


# -------------------------
# DATA FETCH
# -------------------------

def fetch_batch(tickers: list[str]) -> pd.DataFrame:
    """
    Pulls info via yfinance in a robust way:
    - uses yf.Tickers for batching
    - tolerates missing fields
    """
    t = yf.Tickers(" ".join(tickers))
    rows = []

    for tk in tickers:
        try:
            obj = t.tickers.get(tk)
            if obj is None:
                # fallback single
                obj = yf.Ticker(tk)

            info = {}
            try:
                info = obj.fast_info or {}
            except Exception:
                info = {}

            # Full info (can be slow/fail) -> guard hard
            full = {}
            try:
                full = obj.info or {}
            except Exception:
                full = {}

            # Price
            price = safe_float(info.get("last_price")) or safe_float(full.get("currentPrice")) or safe_float(full.get("regularMarketPrice"))
            currency = (info.get("currency") or full.get("currency") or "").strip()

            # Valuation
            pe = safe_float(full.get("trailingPE")) or safe_float(full.get("forwardPE"))

            # Dividend yield:
            # Yahoo: full["dividendYield"] is fraction (0.03 = 3%)
            div_yield = safe_float(full.get("dividendYield"))
            # Some tickers have "yield" or "trailingAnnualDividendYield"
            if div_yield is None:
                div_yield = safe_float(full.get("trailingAnnualDividendYield"))

            # 5y dividend growth:
            # Yahoo doesn't provide a consistent "5y dividend growth" field for all.
            # We'll approximate using dividends history if available.
            div_growth_5y = None
            try:
                divs = obj.dividends
                if divs is not None and len(divs) > 0:
                    # yearly sums for last 6 years to compute 5y CAGR
                    s = divs.copy()
                    s.index = pd.to_datetime(s.index)
                    yearly = s.resample("Y").sum()
                    yearly.index = yearly.index.year
                    if len(yearly) >= 6:
                        last_year = int(yearly.index.max())
                        first_year = last_year - 5
                        a = float(yearly.loc[first_year])
                        b = float(yearly.loc[last_year])
                        if a > 0 and b > 0:
                            div_growth_5y = (b / a) ** (1 / 5) - 1
            except Exception:
                div_growth_5y = None

            name = full.get("shortName") or full.get("longName") or ""
            sector = full.get("sector") or ""
            industry = full.get("industry") or ""
            country = full.get("country") or ""
            exchange = full.get("exchange") or full.get("fullExchangeName") or ""

            rows.append({
                "Ticker": tk,
                "Name": name,
                "Price": price,
                "Currency": currency,
                "PE": pe,
                "DividendYield": div_yield,          # fraction
                "DividendGrowth5Y": div_growth_5y,   # fraction
                "Sector": sector,
                "Industry": industry,
                "Country": country,
                "Exchange": exchange,
            })
        except Exception as e:
            rows.append({
                "Ticker": tk,
                "Name": "",
                "Price": None,
                "Currency": "",
                "PE": None,
                "DividendYield": None,
                "DividendGrowth5Y": None,
                "Sector": "",
                "Industry": "",
                "Country": "",
                "Exchange": "",
                "Error": str(e),
            })

    return pd.DataFrame(rows)


def fetch_all(tickers: list[str]) -> pd.DataFrame:
    all_frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        df = fetch_batch(batch)
        all_frames.append(df)
        if i + BATCH_SIZE < len(tickers):
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# -------------------------
# HTML DASHBOARD
# -------------------------

def render_cell(value: str, css_class: str = "") -> str:
    if value is None:
        return ""
    v = str(value)
    if v.strip() == "":
        return ""
    if css_class:
        return '<span class="pill ' + css_class + '">' + v + '</span>'
    return v


def build_html(df: pd.DataFrame, generated_at: str) -> str:
    # Minimal, fast client-side filtering/search/sort (no external libs).
    css = """
    :root { --bg:#0b0f14; --card:#111826; --text:#e6edf3; --muted:#9aa4af; --line:#223042; }
    body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
           margin:0; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .top { display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .title { font-size: 20px; font-weight: 700; }
    .meta { color: var(--muted); font-size: 12px; }
    .card { margin-top: 14px; background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 14px; }
    input, select { background:#0e1622; border:1px solid var(--line); color:var(--text); border-radius:10px; padding:10px 12px; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    table { width:100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding:10px 10px; border-bottom:1px solid var(--line); vertical-align:middle; }
    th { text-align:left; font-size:12px; color: var(--muted); cursor:pointer; user-select:none; }
    td { font-size:13px; }
    .pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; border:1px solid var(--line); }
    .pill-green { background: rgba(26, 173, 89, 0.18); }
    .pill-amber { background: rgba(255, 200, 64, 0.18); }
    .pill-gray  { background: rgba(148, 163, 184, 0.12); }
    .right { text-align:right; }
    .small { font-size:12px; color:var(--muted); }
    .badge { color: var(--muted); font-size:12px; }
    .row-muted { color: var(--muted); }
    """

    # Prepare rows for JS
    safe_cols = [
        "Ticker","Name","Price","Currency","PE","DividendYield","DividendGrowth5Y",
        "Score","Reco","Indexes","Sector","Country"
    ]
    dd = df.copy()
    for c in safe_cols:
        if c not in dd.columns:
            dd[c] = ""

    # JSON rows for JS rendering
    records = dd[safe_cols].replace({np.nan: ""}).to_dict(orient="records")
    data_json = json.dumps(records)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Dividend Screener</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="title">Dividend Screener (DK + US)</div>
      <div class="meta">Generated: {generated_at}</div>
    </div>
    <div class="controls">
      <input id="q" type="text" placeholder="Search ticker / name / index / sector..." size="34"/>
      <select id="reco">
        <option value="">All</option>
        <option value="BUY">BUY</option>
        <option value="HOLD">HOLD</option>
        <option value="WATCH">WATCH</option>
      </select>
      <select id="idx">
        <option value="">All indexes</option>
      </select>
    </div>
  </div>

  <div class="card">
    <div class="small" id="count"></div>
    <table id="tbl">
      <thead>
        <tr>
          <th data-k="Ticker">Ticker</th>
          <th data-k="Name">Name</th>
          <th class="right" data-k="Price">Price</th>
          <th data-k="Currency">CCY</th>
          <th class="right" data-k="PE">PE</th>
          <th class="right" data-k="DividendYield">Yield</th>
          <th class="right" data-k="DividendGrowth5Y">Div G 5Y</th>
          <th class="right" data-k="Score">Score</th>
          <th data-k="Reco">Reco</th>
          <th data-k="Indexes">Indexes</th>
          <th data-k="Sector">Sector</th>
          <th data-k="Country">Country</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="meta" style="margin-top:12px;">
    Tip: sort by clicking headers. Dashboard is static (GitHub Pages).
  </div>
</div>

<script>
const DATA = {data_json};

function pillClass(label) {{
  const v = (label || "").toUpperCase();
  if (v === "BUY") return "pill-green";
  if (v === "HOLD") return "pill-amber";
  return "pill-gray";
}}

function fmtNum(x, digits=2) {{
  if (x === "" || x === null || x === undefined) return "";
  const n = Number(x);
  if (!isFinite(n)) return "";
  return n.toFixed(digits);
}}

function fmtPct(x, digits=1) {{
  if (x === "" || x === null || x === undefined) return "";
  const n = Number(x);
  if (!isFinite(n)) return "";
  return (n*100).toFixed(digits) + "%";
}}

function uniqueIndexes(data) {{
  const set = new Set();
  data.forEach(r => {{
    const s = (r.Indexes || "").split("|").map(x => x.trim()).filter(Boolean);
    s.forEach(x => set.add(x));
  }});
  return Array.from(set).sort();
}}

const qEl = document.getElementById("q");
const recoEl = document.getElementById("reco");
const idxEl = document.getElementById("idx");
const tbody = document.querySelector("#tbl tbody");
const countEl = document.getElementById("count");

let sortKey = "Score";
let sortDir = "desc";

function rowMatches(r, q, reco, idx) {{
  const hay = (r.Ticker+" "+r.Name+" "+r.Indexes+" "+r.Sector+" "+r.Country).toLowerCase();
  if (q && !hay.includes(q)) return false;
  if (reco && String(r.Reco).toUpperCase() !== reco) return false;
  if (idx) {{
    const s = (r.Indexes || "");
    if (!s.includes(idx)) return false;
  }}
  return true;
}}

function render() {{
  const q = (qEl.value || "").trim().toLowerCase();
  const reco = (recoEl.value || "").trim().toUpperCase();
  const idx = (idxEl.value || "").trim();

  let rows = DATA.filter(r => rowMatches(r, q, reco, idx));

  rows.sort((a,b) => {{
    const av = a[sortKey];
    const bv = b[sortKey];
    const an = Number(av);
    const bn = Number(bv);
    let cmp = 0;
    if (isFinite(an) && isFinite(bn)) cmp = an - bn;
    else cmp = String(av).localeCompare(String(bv));
    return (sortDir === "asc") ? cmp : -cmp;
  }});

  tbody.innerHTML = "";
  rows.forEach(r => {{
    const tr = document.createElement("tr");

    const tds = [
      r.Ticker || "",
      r.Name || "",
      fmtNum(r.Price, 2),
      r.Currency || "",
      fmtNum(r.PE, 1),
      fmtPct(r.DividendYield, 1),
      fmtPct(r.DividendGrowth5Y, 1),
      fmtNum(r.Score, 2),
      r.Reco || "",
      r.Indexes || "",
      r.Sector || "",
      r.Country || ""
    ];

    tds.forEach((v, i) => {{
      const td = document.createElement("td");
      if ([2,4,5,6,7].includes(i)) td.className = "right";
      if (i === 8) {{
        const span = document.createElement("span");
        span.className = "pill " + pillClass(v);
        span.textContent = v;
        td.appendChild(span);
      }} else {{
        td.textContent = v;
      }}
      tr.appendChild(td);
    }});

    tbody.appendChild(tr);
  }});

  countEl.textContent = rows.length + " rows";
}}

function wireSorting() {{
  document.querySelectorAll("th[data-k]").forEach(th => {{
    th.addEventListener("click", () => {{
      const k = th.getAttribute("data-k");
      if (sortKey === k) sortDir = (sortDir === "asc") ? "desc" : "asc";
      else {{
        sortKey = k;
        sortDir = (k === "Ticker" || k === "Name" || k === "Reco" || k === "Indexes" || k === "Sector" || k === "Country")
          ? "asc" : "desc";
      }}
      render();
    }});
  }});
}}

function initIndexFilter() {{
  const opts = uniqueIndexes(DATA);
  opts.forEach(x => {{
    const o = document.createElement("option");
    o.value = x;
    o.textContent = x;
    idxEl.appendChild(o);
  }});
}}

[qEl, recoEl, idxEl].forEach(el => el.addEventListener("input", render));
[qEl, recoEl, idxEl].forEach(el => el.addEventListener("change", render));

initIndexFilter();
wireSorting();
render();
</script>

</body>
</html>
"""
    return html


# -------------------------
# MAIN
# -------------------------

def main() -> int:
    ensure_dirs()

    tickers = read_tickers()
    if not tickers:
        print("No valid tickers found.")
        return 1

    index_map = read_index_map()

    print(f"Tickers: {len(tickers)}")
    df = fetch_all(tickers)

    # Add index exposure column
    df["Indexes"] = df["Ticker"].map(index_map).fillna("")

    # Score + reco
    df["Score"] = df.apply(lambda r: score_row(
        safe_float(r.get("DividendYield")),
        safe_float(r.get("DividendGrowth5Y")),
        safe_float(r.get("PE"))
    ), axis=1)

    df["Reco"] = df.apply(lambda r: recommendation(
        safe_float(r.get("Score")) or 0.0,
        safe_float(r.get("DividendYield")),
        safe_float(r.get("PE"))
    ), axis=1)

    # Sort output
    df = df.sort_values(["Reco", "Score"], ascending=[True, False]).reset_index(drop=True)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {OUT_CSV}")

    # Write HTML dashboard
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(df, generated_at)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote: {OUT_HTML}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
