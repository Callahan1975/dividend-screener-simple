#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (Stable v3) – DK + US
- Reads tickers from tickers.txt (supports comments/sections)
- Fetches data via yfinance
- Calculates yield, 5Y dividend growth (from dividend history), fair value (Yield + DDM), upside, score
- Flags special dividends (high yield) and caps yield for modeling
- Produces docs/index.html dashboard with filters + sorting
- Writes data/screener_results.csv
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
    raise SystemExit("Missing dependency: yfinance. Add it to requirements.txt")

# ----------------------------
# PATHS
# ----------------------------
DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
OUT_CSV = DATA_DIR / "screener_results.csv"
OUT_HTML = DOCS_DIR / "index.html"

# ----------------------------
# SETTINGS (tune later)
# ----------------------------
BATCH_SIZE = 40
SLEEP_BETWEEN_BATCHES = 1.0

# Yield sanity controls
SPECIAL_YIELD_THRESHOLD = 0.12   # >12% = likely special/one-off/reit/mlp etc
YIELD_CAP_FOR_MODEL = 0.12       # cap yield used in fair value/score

# Fair value assumptions
NORMALIZED_YIELD = 0.03          # "fair" market yield for quality dividend payer
DISCOUNT_RATE = 0.08             # DDM discount rate
MAX_DDM_GROWTH = 0.06            # cap long-term growth

# Action thresholds (conservative)
ACTION_BUY_UPSIDE = 15.0         # Upside (Yield) >= 15% => KØB NU (if score OK and not special)
ACTION_EXPENSIVE_UPSIDE = 5.0    # Upside (Yield) < 5%  => FOR DYR
ACTION_MIN_SCORE = 75.0          # minimum score for KØB NU


# ----------------------------
# HELPERS
# ----------------------------
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


def normalize_fraction(v: float | None) -> float | None:
    """
    Yahoo sometimes returns dividendYield as:
      - fraction (0.035)
      - percent (3.5)
    Normalize to fraction.
    """
    if v is None:
        return None
    return v / 100.0 if v > 1 else v


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dedupe_preserve(items: list[str]) -> list[str]:
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
    # reject "sentences"/headers
    if " " in t:
        return False
    # reject markdown-ish separators
    if t in {"---", "END"}:
        return False
    # reject lines that start with non-ticker prefix
    if t.startswith("#"):
        return False
    return True


# ----------------------------
# INPUT
# ----------------------------
def read_tickers() -> tuple[list[str], str]:
    """
    Supports tickers.txt with sections like:

      # --- DK ---
      NOVO-B.CO
      PNDORA.CO

      # --- US ---
      PEP
      KO

    It will ignore all comment/header lines.
    """
    candidates = [Path("tickers.txt"), Path("data") / "tickers.txt", Path("input.csv"), Path("tickers.csv")]
    for p in candidates:
        if not p.exists():
            continue

        if p.suffix.lower() == ".txt":
            lines = p.read_text(encoding="utf-8").splitlines()
            tickers = []
            for raw in lines:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                if not is_valid_ticker(line):
                    continue
                tickers.append(line)
            tickers = dedupe_preserve(tickers)
            if tickers:
                return tickers, str(p)

        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            # prefer Ticker column, else first column
            col = "Ticker" if "Ticker" in df.columns else df.columns[0]
            tickers = [str(x).strip() for x in df[col].tolist()]
            tickers = [t for t in tickers if is_valid_ticker(t)]
            tickers = dedupe_preserve(tickers)
            if tickers:
                return tickers, f"{p} (col={col})"

    raise FileNotFoundError("No tickers input found. Create tickers.txt with one ticker per line.")


def read_index_map() -> dict[str, str]:
    """
    index_map.csv columns:
      Ticker, Indexes
    Indexes can be pipe-separated: "S&P 500|Nasdaq 100|OMXC25"
    """
    for p in [Path("index_map.csv"), Path("data") / "index_map.csv"]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "Ticker" not in df.columns or "Indexes" not in df.columns:
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
    return {}


# ----------------------------
# FINANCE CALCS
# ----------------------------
def dividend_growth_5y_from_history(tkr: yf.Ticker) -> float | None:
    """
    Uses dividend time series -> yearly sums -> 5Y CAGR.
    Returns fraction (0.10 = 10%)
    """
    try:
        divs = tkr.dividends
        if divs is None or len(divs) == 0:
            return None
        s = divs.copy()
        s.index = pd.to_datetime(s.index)
        yearly = s.resample("Y").sum()
        yearly.index = yearly.index.year
        if len(yearly) < 6:
            return None
        last_year = int(yearly.index.max())
        first_year = last_year - 5
        if first_year not in yearly.index:
            return None
        a = float(yearly.loc[first_year])
        b = float(yearly.loc[last_year])
        if a <= 0 or b <= 0:
            return None
        return (b / a) ** (1 / 5) - 1
    except Exception:
        return None


def fair_value_yield(price: float | None, yield_frac: float | None) -> float | None:
    if price is None or yield_frac is None:
        return None
    if price <= 0 or yield_frac <= 0:
        return None
    annual_div = price * yield_frac
    return round(annual_div / NORMALIZED_YIELD, 2)


def fair_value_ddm(price: float | None, yield_frac: float | None, growth: float | None) -> float | None:
    if price is None or yield_frac is None:
        return None
    if price <= 0 or yield_frac <= 0:
        return None
    g = growth if growth is not None else 0.02
    g = min(g, MAX_DDM_GROWTH)
    r = DISCOUNT_RATE
    if r <= g:
        return None
    annual_div = price * yield_frac
    d1 = annual_div * (1 + g)
    return round(d1 / (r - g), 2)


def upside_pct(price: float | None, fair: float | None) -> float | None:
    if price is None or fair is None or price <= 0:
        return None
    return round((fair / price - 1) * 100, 1)


def score_row(yield_frac: float | None, growth: float | None, pe: float | None) -> float:
    # yield: 0% -> 0, 6% -> 100
    s_y = 0.0 if yield_frac is None else clamp((yield_frac / 0.06) * 100.0, 0.0, 100.0)
    # growth: 0% -> 0, 15% -> 100
    s_g = 0.0 if growth is None else clamp((growth / 0.15) * 100.0, 0.0, 100.0)
    # valuation: PE 10 -> 100, PE 30 -> 0
    if pe is None or pe <= 0:
        s_v = 40.0
    else:
        s_v = clamp((30.0 - pe) / 20.0 * 100.0, 0.0, 100.0)

    score = 0.35 * s_y + 0.35 * s_g + 0.30 * s_v
    return float(round(score, 2))


def reco_label(score: float, is_special: bool) -> str:
    if is_special:
        return "WATCH"
    if score >= 75:
        return "BUY"
    if score >= 55:
        return "HOLD"
    return "WATCH"


def action_label(upside_y: float | None, score: float | None, is_special: bool) -> str:
    if is_special:
        return "VENT"
    if upside_y is None or score is None:
        return "VENT"
    if upside_y >= ACTION_BUY_UPSIDE and score >= ACTION_MIN_SCORE:
        return "KØB NU"
    if upside_y < ACTION_EXPENSIVE_UPSIDE:
        return "FOR DYR"
    return "VENT"


# ----------------------------
# FETCH DATA
# ----------------------------
def fetch_batch(tickers: list[str]) -> pd.DataFrame:
    yt = yf.Tickers(" ".join(tickers))
    rows = []

    for tk in tickers:
        try:
            obj = yt.tickers.get(tk) or yf.Ticker(tk)
            info = obj.info or {}

            price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            currency = (info.get("currency") or "").strip()
            pe = safe_float(info.get("trailingPE") or info.get("forwardPE"))

            # dividend yield robust:
            # prefer annual dividend rate / price when available
            annual_div = safe_float(info.get("trailingAnnualDividendRate") or info.get("dividendRate"))
            y = None
            if price and annual_div and annual_div > 0:
                y = annual_div / price
            else:
                y = safe_float(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))

            y = normalize_fraction(y)

            g5 = dividend_growth_5y_from_history(obj)  # fraction

            sector = info.get("sector") or ""
            country = info.get("country") or ""

            rows.append({
                "Ticker": tk,
                "Name": info.get("shortName") or info.get("longName") or "",
                "Price": price,
                "Currency": currency,
                "PE": pe,
                "DividendYield": y,
                "DividendGrowth5Y": g5,
                "Sector": sector,
                "Country": country,
            })

        except Exception as e:
            # IMPORTANT: never fail the whole run
            rows.append({
                "Ticker": tk,
                "Name": "",
                "Price": None,
                "Currency": "",
                "PE": None,
                "DividendYield": None,
                "DividendGrowth5Y": None,
                "Sector": "",
                "Country": "",
                "Error": str(e),
            })

    return pd.DataFrame(rows)


def fetch_all(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        frames.append(fetch_batch(batch))
        if i + BATCH_SIZE < len(tickers):
            time.sleep(SLEEP_BETWEEN_BATCHES)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------
# HTML DASHBOARD
# ----------------------------
def build_html(df: pd.DataFrame, generated_at: str, source: str) -> str:
    css = """
    :root { --bg:#0b0f14; --card:#111826; --text:#e6edf3; --muted:#9aa4af; --line:#223042; }
    body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1800px; margin: 0 auto; padding: 20px; }
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
        "YieldPct","DivG5YPct",
        "SpecialDividend",
        "FairValue_Yield","FairValue_DDM",
        "Upside_Yield_%","Upside_DDM_%",
        "Score","Reco","Action",
        "Indexes","Sector","Country"
    ]

    dd = df.copy()
    for c in cols:
        if c not in dd.columns:
            dd[c] = ""

    data = dd[cols].replace({np.nan: ""}).to_dict(orient="records")
    data_json = json.dumps(data)

    # IMPORTANT: no f-string with JS braces; we insert placeholders safely
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Dividend Screener</title>
<style>__CSS__</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="title">Dividend Screener (DK + US)</div>
      <div class="meta">Generated: __GEN__ • Source: __SRC__</div>
    </div>
    <div class="controls">
      <input id="q" type="text" placeholder="Search ticker / name / index / sector..." size="34"/>

      <select id="action">
        <option value="">All actions</option>
        <option value="KØB NU">KØB NU</option>
        <option value="VENT">VENT</option>
        <option value="FOR DYR">FOR DYR</option>
      </select>

      <select id="reco">
        <option value="">All reco</option>
        <option value="BUY">BUY</option>
        <option value="HOLD">HOLD</option>
        <option value="WATCH">WATCH</option>
      </select>

      <select id="special">
        <option value="">All</option>
        <option value="true">Special only</option>
        <option value="false">No special</option>
      </select>

      <select id="sector"><option value="">All sectors</option></select>
      <select id="country"><option value="">All countries</option></select>
      <select id="idx"><option value="">All indexes</option></select>
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
          <th class="right" data-k="YieldPct">Yield</th>
          <th class="right" data-k="DivG5YPct">Div G 5Y</th>
          <th data-k="SpecialDividend">Special</th>
          <th class="right" data-k="FairValue_Yield">Fair Value (Yield)</th>
          <th class="right" data-k="FairValue_DDM">Fair Value (DDM)</th>
          <th class="right" data-k="Upside_Yield_%">Upside (Yield) %</th>
          <th class="right" data-k="Upside_DDM_%">Upside (DDM) %</th>
          <th class="right" data-k="Score">Score</th>
          <th data-k="Reco">Reco</th>
          <th data-k="Action">Action</th>
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
const DATA = __DATA__;

function pillClass(label) {
  const v = (label || "").toUpperCase();
  if (v === "BUY" || v === "KØB NU") return "pill-green";
  if (v === "HOLD" || v === "VENT") return "pill-amber";
  return "pill-gray";
}

function uniqueValues(key) {
  const set = new Set();
  DATA.forEach(r => {
    const v = (r[key] || "").toString().trim();
    if (v) set.add(v);
  });
  return Array.from(set).sort();
}

function uniqueIndexes() {
  const set = new Set();
  DATA.forEach(r => {
    const s = (r.Indexes || "").split("|").map(x => x.trim()).filter(Boolean);
    s.forEach(x => set.add(x));
  });
  return Array.from(set).sort();
}

const qEl = document.getElementById("q");
const actionEl = document.getElementById("action");
const recoEl = document.getElementById("reco");
const specialEl = document.getElementById("special");
const sectorEl = document.getElementById("sector");
const countryEl = document.getElementById("country");
const idxEl = document.getElementById("idx");
const tbody = document.querySelector("#tbl tbody");
const countEl = document.getElementById("count");

let sortKey = "Score";
let sortDir = "desc";

function rowMatches(r, q, action, reco, special, sector, country, idx) {
  const hay = (r.Ticker+" "+r.Name+" "+r.Indexes+" "+r.Sector+" "+r.Country).toLowerCase();
  if (q && !hay.includes(q)) return false;
  if (action && String(r.Action) !== action) return false;
  if (reco && String(r.Reco).toUpperCase() !== reco) return false;

  if (special !== "") {
    const want = (special === "true");
    const isSpec = String(r.SpecialDividend).toLowerCase() === "true";
    if (want !== isSpec) return false;
  }

  if (sector && String(r.Sector || "") !== sector) return false;
  if (country && String(r.Country || "") !== country) return false;

  if (idx) {
    const s = (r.Indexes || "");
    if (!s.includes(idx)) return false;
  }

  return true;
}

function render() {
  const q = (qEl.value || "").trim().toLowerCase();
  const action = (actionEl.value || "").trim();
  const reco = (recoEl.value || "").trim().toUpperCase();
  const special = (specialEl.value || "").trim();
  const sector = (sectorEl.value || "").trim();
  const country = (countryEl.value || "").trim();
  const idx = (idxEl.value || "").trim();

  let rows = DATA.filter(r => rowMatches(r, q, action, reco, special, sector, country, idx));

  rows.sort((a,b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const an = Number(av);
    const bn = Number(bv);
    let cmp = 0;
    if (isFinite(an) && isFinite(bn)) cmp = an - bn;
    else cmp = String(av).localeCompare(String(bv));
    return (sortDir === "asc") ? cmp : -cmp;
  });

  tbody.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");

    const spec = (String(r.SpecialDividend).toLowerCase() === "true") ? "YES" : "";
    const recoPill = `<span class="pill ${pillClass(r.Reco)}">${r.Reco || ""}</span>`;
    const actionPill = `<span class="pill ${pillClass(r.Action)}">${r.Action || ""}</span>`;

    tr.innerHTML = `
      <td>${r.Ticker||""}</td>
      <td>${r.Name||""}</td>
      <td class="right">${r.Price||""}</td>
      <td>${r.Currency||""}</td>
      <td class="right">${r.PE||""}</td>
      <td class="right">${r.YieldPct||""}</td>
      <td class="right">${r.DivG5YPct||""}</td>
      <td>${spec}</td>
      <td class="right">${r.FairValue_Yield||""}</td>
      <td class="right">${r.FairValue_DDM||""}</td>
      <td class="right">${r["Upside_Yield_%"]||""}</td>
      <td class="right">${r["Upside_DDM_%"]||""}</td>
      <td class="right">${r.Score||""}</td>
      <td>${recoPill}</td>
      <td>${actionPill}</td>
      <td>${r.Indexes||""}</td>
      <td>${r.Sector||""}</td>
      <td>${r.Country||""}</td>
    `;
    tbody.appendChild(tr);
  });

  countEl.textContent = rows.length + " rows";
}

document.querySelectorAll("th[data-k]").forEach(th => {
  th.addEventListener("click", () => {
    const k = th.getAttribute("data-k");
    if (sortKey === k) sortDir = (sortDir === "asc") ? "desc" : "asc";
    else {
      sortKey = k;
      sortDir = (k === "Ticker" || k === "Name" || k === "Reco" || k === "Action" || k === "Indexes" || k === "Sector" || k === "Country" || k === "Currency" || k === "SpecialDividend")
        ? "asc" : "desc";
    }
    render();
  });
});

uniqueIndexes().forEach(x => {
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  idxEl.appendChild(o);
});

uniqueValues("Sector").forEach(x => {
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  sectorEl.appendChild(o);
});

uniqueValues("Country").forEach(x => {
  const o = document.createElement("option");
  o.value = x;
  o.textContent = x;
  countryEl.appendChild(o);
});

[qEl, actionEl, recoEl, specialEl, sectorEl, countryEl, idxEl].forEach(el => el.addEventListener("input", render));
[qEl, actionEl, recoEl, specialEl, sectorEl, countryEl, idxEl].forEach(el => el.addEventListener("change", render));

render();
</script>
</body>
</html>
"""
    return (
        html.replace("__CSS__", css)
            .replace("__GEN__", generated_at)
            .replace("__SRC__", source)
            .replace("__DATA__", data_json)
    )


# ----------------------------
# MAIN
# ----------------------------
def main() -> int:
    ensure_dirs()

    tickers, source = read_tickers()
    index_map = read_index_map()

    print(f"Loaded {len(tickers)} tickers from {source}")

    df = fetch_all(tickers)

    # Attach indexes
    df["Indexes"] = df["Ticker"].map(index_map).fillna("")

    # Special flag + yield cap for model
    df["SpecialDividend"] = df["DividendYield"].apply(
        lambda y: True if (safe_float(y) is not None and safe_float(y) > SPECIAL_YIELD_THRESHOLD) else False
    )
    df["YieldForModel"] = df["DividendYield"].apply(
        lambda y: min(safe_float(y), YIELD_CAP_FOR_MODEL) if safe_float(y) is not None else None
    )

    # Derived display columns (percent strings)
    df["YieldPct"] = df["DividendYield"].apply(lambda y: "" if y is None else f"{y*100:.1f}%")
    df["DivG5YPct"] = df["DividendGrowth5Y"].apply(lambda g: "" if g is None else f"{g*100:.1f}%")

    # Fair values + upside
    df["FairValue_Yield"] = df.apply(lambda r: fair_value_yield(safe_float(r["Price"]), safe_float(r["YieldForModel"])), axis=1)
    df["FairValue_DDM"] = df.apply(lambda r: fair_value_ddm(safe_float(r["Price"]), safe_float(r["YieldForModel"]), safe_float(r["DividendGrowth5Y"])), axis=1)

    df["Upside_Yield_%"] = df.apply(lambda r: upside_pct(safe_float(r["Price"]), safe_float(r["FairValue_Yield"])), axis=1)
    df["Upside_DDM_%"] = df.apply(lambda r: upside_pct(safe_float(r["Price"]), safe_float(r["FairValue_DDM"])), axis=1)

    # Score / reco / action
    df["Score"] = df.apply(lambda r: score_row(safe_float(r["YieldForModel"]), safe_float(r["DividendGrowth5Y"]), safe_float(r["PE"])), axis=1)
    df["Reco"] = df.apply(lambda r: reco_label(float(r["Score"] or 0), bool(r["SpecialDividend"])), axis=1)
    df["Action"] = df.apply(lambda r: action_label(safe_float(r["Upside_Yield_%"]), safe_float(r["Score"]), bool(r["SpecialDividend"])), axis=1)

    # Save CSV + HTML
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(df, generated_at, source), encoding="utf-8")

    print(f"Wrote {OUT_CSV} and {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
