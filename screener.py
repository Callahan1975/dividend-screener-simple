#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) - GitHub Actions + GitHub Pages

INPUT (auto-detected, tickers.txt preferred):
  - tickers.txt              (root)  one ticker per line   (PREFERRED)
  - data/tickers.txt         (one ticker per line)
  - data/tickers.csv         (Ticker column OR best column)
  - data/input.csv           (Ticker column OR best column)
  - input.csv                (root, Ticker column OR best column)
  - tickers.csv              (root, Ticker column OR best column)

Optional index mapping:
  - index_map.csv (root) or data/index_map.csv
    columns: Ticker, Indexes
    Indexes can be pipe-separated: "S&P 500|Nasdaq 100|OMXC25"

OUTPUT:
  - data/screener_results.csv
  - docs/index.html
"""

from __future__ import annotations

import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Ensure requirements.txt installs yfinance pandas numpy")
    raise


# -------------------------
# PATHS / CONFIG
# -------------------------

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")

INPUT_CANDIDATES = [
    (Path("tickers.txt"), "txt"),
    (DATA_DIR / "tickers.txt", "txt"),
    (DATA_DIR / "tickers.csv", "csv"),
    (DATA_DIR / "input.csv", "csv"),
    (Path("input.csv"), "csv"),
    (Path("tickers.csv"), "csv"),
]

INDEX_MAP_CANDIDATES = [
    Path("index_map.csv"),
    DATA_DIR / "index_map.csv",
]

OUT_CSV = DATA_DIR / "screener_results.csv"
OUT_HTML = DOCS_DIR / "index.html"

BATCH_SIZE = 40
SLEEP_BETWEEN_BATCHES_SEC = 1.0

# Scoring weights
WEIGHT_YIELD = 0.35
WEIGHT_GROWTH = 0.35
WEIGHT_VALUATION = 0.30

# Yield sanity controls
SPECIAL_YIELD_THRESHOLD = 0.12   # >12% treated as likely special/non-repeatable
YIELD_CAP_FOR_MODEL = 0.12       # cap yield used for scoring & fair value

DEFAULT_NORMALIZED_YIELD = 0.03  # used for Fair Value (Yield)
DEFAULT_DISCOUNT_RATE = 0.08     # used for DDM


# -------------------------
# UTILS
# -------------------------

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if pd.isna(x):
                return None
            return float(x)
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def is_valid_ticker(t: str) -> bool:
    if t is None:
        return False
    t = str(t).strip()
    if not t:
        return False
    if t.lower() in {"nan", "none", "null"}:
        return False
    if t.startswith("."):
        return False
    if t.upper() in {".CO", ".TO", ".ST"}:
        return False
    if t.endswith(".CO") and len(t) <= 3:
        return False
    return True


def _best_ticker_column(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return df.columns[0]

    scores = {}
    for c in df.columns:
        vals = [str(x).strip() for x in df[c].tolist()]
        vals = [v for v in vals if is_valid_ticker(v)]
        scores[c] = len(vals)

    best = max(scores, key=scores.get) if scores else df.columns[0]
    best_score = scores.get(best, 0)

    preferred = ["Ticker", "Symbol", "Yahoo", "YahooTicker", "ticker", "symbol"]
    tied = [c for c, s in scores.items() if s == best_score]
    for p in preferred:
        for c in tied:
            if str(c).strip() == p:
                return c

    return best


def detect_and_read_tickers() -> tuple[list[str], str]:
    for path, kind in INPUT_CANDIDATES:
        if not path.exists():
            continue

        if kind == "txt":
            raw = path.read_text(encoding="utf-8").splitlines()
            tickers = []
            for line in raw:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                tickers.append(line)

            tickers = [t for t in tickers if is_valid_ticker(t)]
            tickers = dedupe_preserve_order(tickers)
            if tickers:
                return tickers, str(path)

        if kind == "csv":
            df = pd.read_csv(path)
            col = "Ticker" if "Ticker" in df.columns else _best_ticker_column(df)
            tickers = [str(x).strip() for x in df[col].tolist()]
            tickers = [t for t in tickers if is_valid_ticker(t)]
            tickers = dedupe_preserve_order(tickers)
            if tickers:
                return tickers, f"{path} (col={col})"

    raise FileNotFoundError("No tickers input found. Create tickers.txt (recommended).")


def read_index_map() -> dict[str, str]:
    file_path = None
    for p in INDEX_MAP_CANDIDATES:
        if p.exists():
            file_path = p
            break
    if file_path is None:
        return {}

    df = pd.read_csv(file_path)
    if "Ticker" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
    if "Ticker" not in df.columns:
        return {}

    if "Indexes" not in df.columns:
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
        idx = idx.replace(";", "|").replace(",", "|")
        idx = " | ".join([x.strip() for x in idx.split("|") if x.strip()])
        out[t] = idx
    return out


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_fraction(v: float | None) -> float | None:
    """
    Fix Yahoo inconsistencies:
      - if v is fraction (0.05) keep it
      - if v is percent (5 or 7.41) convert -> 0.05 / 0.0741
    """
    if v is None:
        return None
    if v > 1:
        return v / 100.0
    return v


# -------------------------
# SCORING + FAIR VALUE
# -------------------------

def score_row(div_yield: float | None, div_growth_5y: float | None, pe: float | None) -> float:
    # Yield: 0% -> 0, 6% -> 100 cap
    s_y = 0.0 if div_yield is None else clamp((div_yield / 0.06) * 100.0, 0.0, 100.0)

    # Growth: 0% -> 0, 15% -> 100 cap
    s_g = 0.0 if div_growth_5y is None else clamp((div_growth_5y / 0.15) * 100.0, 0.0, 100.0)

    # Valuation: PE 10 -> 100, PE 30 -> 0
    if pe is None or pe <= 0:
        s_v = 40.0
    else:
        s_v = clamp((30.0 - pe) / (30.0 - 10.0) * 100.0, 0.0, 100.0)

    score = WEIGHT_YIELD * s_y + WEIGHT_GROWTH * s_g + WEIGHT_VALUATION * s_v
    return float(round(score, 2))


def recommendation(score: float, div_yield: float | None, is_special: bool) -> str:
    # If special dividend flagged, do not auto-BUY
    if is_special:
        return "WATCH"
    if score >= 75 and (div_yield is None or div_yield >= 0.015):
        return "BUY"
    if score >= 55:
        return "HOLD"
    return "WATCH"


def fair_value_yield(price: float | None, dividend_yield: float | None, normalized_yield: float = DEFAULT_NORMALIZED_YIELD) -> float | None:
    if price is None or dividend_yield is None:
        return None
    if price <= 0 or dividend_yield <= 0 or normalized_yield <= 0:
        return None
    annual_dividend = price * dividend_yield
    return round(annual_dividend / normalized_yield, 2)


def fair_value_ddm(price: float | None, dividend_yield: float | None, growth: float | None, discount_rate: float = DEFAULT_DISCOUNT_RATE) -> float | None:
    if price is None or dividend_yield is None:
        return None
    if price <= 0 or dividend_yield <= 0:
        return None

    g = growth if growth is not None else 0.02
    g = min(g, 0.06)  # cap long-term growth
    r = discount_rate
    if r <= g:
        return None

    annual_dividend = price * dividend_yield
    d1 = annual_dividend * (1 + g)
    return round(d1 / (r - g), 2)


def upside_pct(price: float | None, fair_value: float | None) -> float | None:
    if price is None or fair_value is None:
        return None
    if price <= 0:
        return None
    return round((fair_value / price - 1) * 100, 1)


# -------------------------
# DATA FETCH (yfinance)
# -------------------------

def fetch_batch(tickers: list[str]) -> pd.DataFrame:
    t = yf.Tickers(" ".join(tickers))
    rows = []

    for tk in tickers:
        try:
            obj = t.tickers.get(tk) or yf.Ticker(tk)

            try:
                fast = obj.fast_info or {}
            except Exception:
                fast = {}

            try:
                info = obj.info or {}
            except Exception:
                info = {}

            price = safe_float(fast.get("last_price")) or safe_float(info.get("currentPrice")) or safe_float(info.get("regularMarketPrice"))
            currency = (fast.get("currency") or info.get("currency") or "").strip()

            pe = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))

            # Robust dividend yield:
            # Prefer annual dividend / price when possible, else use Yahoo yield fields
            annual_div = safe_float(info.get("trailingAnnualDividendRate")) or safe_float(info.get("forwardAnnualDividendRate"))

            div_yield = None
            if price is not None and price > 0 and annual_div is not None and annual_div > 0:
                div_yield = annual_div / price
            else:
                div_yield = safe_float(info.get("dividendYield"))
                if div_yield is None:
                    div_yield = safe_float(info.get("trailingAnnualDividendYield"))

            div_yield = _normalize_fraction(div_yield)

            # 5Y dividend growth from dividend history (CAGR)
            div_growth_5y = None
            try:
                divs = obj.dividends
                if divs is not None and len(divs) > 0:
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

            # normalize if ever returned as percent
            if div_growth_5y is not None and abs(div_growth_5y) > 1:
                div_growth_5y = div_growth_5y / 100.0

            rows.append({
                "Ticker": tk,
                "Name": info.get("shortName") or info.get("longName") or "",
                "Price": price,
                "Currency": currency,
                "PE": pe,
                "DividendYield": div_yield,             # fraction, e.g. 0.035
                "DividendGrowth5Y": div_growth_5y,      # fraction, e.g. 0.08
                "Sector": info.get("sector") or "",
                "Industry": info.get("industry") or "",
                "Country": info.get("country") or "",
                "Exchange": info.get("exchange") or info.get("fullExchangeName") or "",
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
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        frames.append(fetch_batch(batch))
        if i + BATCH_SIZE < len(tickers):
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -------------------------
# HTML DASHBOARD
# -------------------------

def build_html(df: pd.DataFrame, generated_at: str, source: str) -> str:
    css = """
    :root { --bg:#0b0f14; --card:#111826; --text:#e6edf3; --muted:#9aa4af; --line:#223042; }
    body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1700px; margin: 0 auto; padding: 20px; }
    .top { display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .title { font-size: 20px; font-weight: 700; }
    .meta { color: var(--muted); font-size: 12px; }
    .card { margin-top: 14px; background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 14px; }
    input, select { background:#0e1622; border:1px solid var(--line); color:var(--text); border-radius:10px; padding:10px 12px; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    table { width:100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding:10px 10px; border-bottom:1px solid var(--line); vertical-align:middle; }
    th { text-align:left; font-size:12px; color: var(--muted); cursor:pointer; user-select:none; white-space:nowrap; }
    td { font-size:13px; }
    .pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; border:1px solid var(--line); }
    .pill-green { background: rgba(26, 173, 89, 0.18); }
    .pill-amber { background: rgba(255, 200, 64, 0.18); }
    .pill-gray  { background: rgba(148, 163, 184, 0.12); }
    .right { text-align:right; }
    .small { font-size:12px; color:var(--muted); }
    """

    cols = [
        "Ticker","Name","Price","Currency","PE",
        "DividendYield","DividendGrowth5Y",
        "SpecialDividend",
        "FairValue_Yield","FairValue_DDM",
        "Upside_Yield_%","Upside_DDM_%",
        "Score","Reco","Indexes","Sector","Country"
    ]

    dd = df.copy()
    for c in cols:
        if c not in dd.columns:
            dd[c] = ""

    records = dd[cols].replace({np.nan: ""}).to_dict(orient="records")
    data_json = json.dumps(records)

    return f"""<!doctype html>
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
      <div class="meta">Generated: {generated_at} • Source: {source}</div>
    </div>
    <div class="controls">
      <input id="q" type="text" placeholder="Search ticker / name / index / sector..." size="34"/>

      <select id="reco">
        <option value="">All</option>
        <option value="BUY">BUY</option>
        <option value="HOLD">HOLD</option>
        <option value="WATCH">WATCH</option>
      </select>

      <select id="special">
        <option value="">All</option>
        <option value="true">Special only</option>
        <option value="false">No special</option>
      </select>

      <select id="sector">
        <option value="">All sectors</option>
      </select>

      <select id="country">
        <option value="">All countries</option>
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
          <th data-k="SpecialDividend">Special</th>
          <th class="right" data-k="FairValue_Yield">Fair Value (Yield)</th>
          <th class="right" data-k="FairValue_DDM">Fair Value (DDM)</th>
          <th class="right" data-k="Upside_Yield_%">Upside (Yield) %</th>
          <th class="right" data-k="Upside_DDM_%">Upside (DDM) %</th>
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

function uniqueValues(key) {{
  const set = new Set();
  DATA.forEach(r => {{
    const v = (r[key] || "").toString().trim();
    if (v) set.add(v);
  }});
  return Array.from(set).sort();
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
const specialEl = document.getElementById("special");
const sectorEl = document.getElementById("sector");
const countryEl = document.getElementById("country");
const idxEl = document.getElementById("idx");
const tbody = document.querySelector("#tbl tbody");
const countEl = document.getElementById("count");

let sortKey = "Score";
let sortDir = "desc";

function rowMatches(r, q, reco, special, sector, country, idx) {{
  const hay = (r.Ticker+" "+r.Name+" "+r.Indexes+" "+r.Sector+" "+r.Country).toLowerCase();
  if (q && !hay.includes(q)) return false;
  if (reco && String(r.Reco).toUpperCase() !== reco) return false;

  if (special !== "") {{
    const want = (special === "true");
    const isSpec = String(r.SpecialDividend).toLowerCase() === "true";
    if (want !== isSpec) return false;
  }}

  if (sector && String(r.Sector || "") !== sector) return false;
  if (country && String(r.Country || "") !== country) return false;

  if (idx) {{
    const s = (r.Indexes || "");
    if (!s.includes(idx)) return false;
  }}

  return true;
}}

function render() {{
  const q = (qEl.value || "").trim().toLowerCase();
  const reco = (recoEl.value || "").trim().toUpperCase();
  const special = (specialEl.value || "").trim();
  const sector = (sectorEl.value || "").trim();
  const country = (countryEl.value || "").trim();
  const idx = (idxEl.value || "").trim();

  let rows = DATA.filter(r => rowMatches(r, q, reco, special, sector, country, idx));

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

    const spec = (String(r.SpecialDividend).toLowerCase() === "true") ? "YES" : "";

    const upsideYield = (r["Upside_Yield_%"] === "" || r["Upside_Yield_%"] === null || r["Upside_Yield_%"] === undefined)
      ? ""
      : fmtNum(r["Upside_Yield_%"], 1) + "%";

    const upsideDDM = (r["Upside_DDM_%"] === "" || r["Upside_DDM_%"] === null || r["Upside_DDM_%"] === undefined)
      ? ""
      : fmtNum(r["Upside_DDM_%"], 1) + "%";

    const tds = [
      r.Ticker || "",
      r.Name || "",
      fmtNum(r.Price, 2),
      r.Currency || "",
      fmtNum(r.PE, 1),
      fmtPct(r.DividendYield, 1),
      fmtPct(r.DividendGrowth5Y, 1),
      spec,
      fmtNum(r.FairValue_Yield, 2),
      fmtNum(r.FairValue_DDM, 2),
      upsideYield,
      upsideDDM,
      fmtNum(r.Score, 2),
      r.Reco || "",
      r.Indexes || "",
      r.Sector || "",
      r.Country || ""
    ];

    tds.forEach((v, i) => {{
      const td = document.createElement("td");
      if ([2,4,5,6,8,9,10,11,12].includes(i)) td.className = "right";
      if (i === 13) {{
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

document.querySelectorAll("th[data-k]").forEach(th => {{
  th.addEventListener("click", () => {{
    const k = th.getAttribute("data-k");
    if (sortKey === k) sortDir = (sortDir === "asc") ? "desc" : "asc";
    else {{
      sortKey = k;
      sortDir = (k === "Ticker" || k === "Name" || k === "Reco" || k === "Indexes" || k === "Sector" || k === "Country" || k === "Currency" || k === "SpecialDividend")
        ? "asc" : "desc";
    }}
    render();
  }});
}});

uniqueIndexes(DATA).forEach(x => {{
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  idxEl.appendChild(o);
}});

uniqueValues("Sector").forEach(x => {{
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  sectorEl.appendChild(o);
}});

uniqueValues("Country").forEach(x => {{
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  countryEl.appendChild(o);
}});

[qEl, recoEl, specialEl, sectorEl, countryEl, idxEl].forEach(el => el.addEventListener("input", render));
[qEl, recoEl, specialEl, sectorEl, countryEl, idxEl].forEach(el => el.addEventListener("change", render));

render();
</script>
</body>
</html>
"""


# -------------------------
# MAIN
# -------------------------

def main() -> int:
    ensure_dirs()

    tickers, source = detect_and_read_tickers()
    print(f"Tickers loaded: {len(tickers)} from {source}")

    index_map = read_index_map()

    df = fetch_all(tickers)

    # Index exposure
    df["Indexes"] = df["Ticker"].map(index_map).fillna("")

    # Special dividend flag (based on raw yield)
    df["SpecialDividend"] = df["DividendYield"].apply(
        lambda y: True if (safe_float(y) is not None and safe_float(y) > SPECIAL_YIELD_THRESHOLD) else False
    )

    # Yield to use in models (cap extreme values)
    df["YieldForModel"] = df["DividendYield"].apply(
        lambda y: min(safe_float(y), YIELD_CAP_FOR_MODEL) if safe_float(y) is not None else None
    )

    # Score + reco (use capped yield, and demote specials)
    df["Score"] = df.apply(lambda r: score_row(
        safe_float(r.get("YieldForModel")),
        safe_float(r.get("DividendGrowth5Y")),
        safe_float(r.get("PE"))
    ), axis=1)

    df["Reco"] = df.apply(lambda r: recommendation(
        float(r.get("Score") or 0.0),
        safe_float(r.get("YieldForModel")),
        bool(r.get("SpecialDividend"))
    ), axis=1)

    # Fair Values + Upside (use capped yield)
    df["FairValue_Yield"] = df.apply(lambda r: fair_value_yield(
        safe_float(r.get("Price")),
        safe_float(r.get("YieldForModel")),
        DEFAULT_NORMALIZED_YIELD
    ), axis=1)

    df["FairValue_DDM"] = df.apply(lambda r: fair_value_ddm(
        safe_float(r.get("Price")),
        safe_float(r.get("YieldForModel")),
        safe_float(r.get("DividendGrowth5Y")),
        DEFAULT_DISCOUNT_RATE
    ), axis=1)

    df["Upside_Yield_%"] = df.apply(lambda r: upside_pct(
        safe_float(r.get("Price")),
        safe_float(r.get("FairValue_Yield"))
    ), axis=1)

    df["Upside_DDM_%"] = df.apply(lambda r: upside_pct(
        safe_float(r.get("Price")),
        safe_float(r.get("FairValue_DDM"))
    ), axis=1)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Reco", "Score"], ascending=[True, False]).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {OUT_CSV}")

    # Save HTML
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(df, generated_at, source), encoding="utf-8")
    print(f"Wrote: {OUT_HTML}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
