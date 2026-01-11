from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# =========================
# Settings
# =========================
DEFAULT_TARGET_YIELD_PCT = 2.75   # fallback if we can't get/estimate a target yield
ENTRY_DISCOUNT_PCT = 10.0         # Entry = Fair * (1 - discount)
TARGET_YIELD_MIN_PCT = 1.0
TARGET_YIELD_MAX_PCT = 12.0

# Optional index mapping file in repo root
# Format:
# ticker,SP500,NASDAQ100,OMXC25
# AAPL,1,1,0
INDEX_MAP_FILENAME = "index_map.csv"

# Paths
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "input.csv"
DOCS_DIR = ROOT / "docs"
DOCS_INDEX = DOCS_DIR / "index.html"
DOCS_OUTCSV = DOCS_DIR / "output.csv"
INDEX_MAP_PATH = ROOT / INDEX_MAP_FILENAME


# =========================
# Helpers
# =========================
def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def read_tickers() -> list[str]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Create input.csv with a column named 'ticker'.")

    df = pd.read_csv(INPUT_CSV)
    if "ticker" not in df.columns:
        raise ValueError("input.csv must contain a column named 'ticker'")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Deduplicate keep order
    seen = set()
    out = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_live_price(t: yf.Ticker, info: dict) -> float | None:
    # 1) fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            p = fi.get("last_price") or fi.get("lastPrice")
            p = safe_float(p)
            if p and p > 0:
                return p
    except Exception:
        pass

    # 2) info
    p = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    if p and p > 0:
        return p

    # 3) last close
    try:
        hist = t.history(period="5d", interval="1d")
        if hist is not None and not hist.empty:
            p = safe_float(hist["Close"].iloc[-1])
            if p and p > 0:
                return p
    except Exception:
        pass

    return None


def trailing_12m_dividend(t: yf.Ticker) -> float | None:
    try:
        div = t.dividends
        if div is None or div.empty:
            return None
        div = div.sort_index()
        cutoff = div.index.max() - pd.Timedelta(days=365)
        d12 = div[div.index > cutoff]
        s = float(d12.sum())
        return s if s > 0 else None
    except Exception:
        return None


def normalize_target_yield(target_yield_pct: float | None, current_yield_pct: float | None) -> float | None:
    if target_yield_pct is None and current_yield_pct is None:
        return DEFAULT_TARGET_YIELD_PCT

    ty = target_yield_pct if target_yield_pct is not None else current_yield_pct
    if ty is None:
        ty = DEFAULT_TARGET_YIELD_PCT

    # Clamp vs current yield (avoid "PEP is worth 2x" situations)
    if current_yield_pct is not None and current_yield_pct > 0:
        lo = 0.60 * current_yield_pct
        hi = 1.40 * current_yield_pct
        ty = max(lo, min(hi, ty))

    ty = max(TARGET_YIELD_MIN_PCT, min(TARGET_YIELD_MAX_PCT, ty))
    return ty


def calc_fair_and_entry(div_rate: float | None, target_yield_pct: float | None) -> tuple[float | None, float | None]:
    if div_rate is None or target_yield_pct is None or target_yield_pct <= 0:
        return None, None
    fair = div_rate / (target_yield_pct / 100.0)
    entry = fair * (1.0 - ENTRY_DISCOUNT_PCT / 100.0)
    return round(fair, 2), round(entry, 2)


def pct_diff(target_price: float | None, price: float | None) -> float | None:
    if target_price is None or price is None or price <= 0:
        return None
    return round((target_price / price - 1.0) * 100.0, 1)


def rating_from_entry(price: float | None, entry: float | None, fair: float | None) -> str:
    if price is None or entry is None or fair is None:
        return "HOLD"
    if price <= entry:
        return "BUY"
    if price <= fair:
        return "HOLD"
    return "SELL"


def quality_score(div_yield_pct: float | None, payout_pct: float | None, pe_ttm: float | None, to_entry_pct: float | None) -> int:
    score = 50

    # Yield (neutral, not too aggressive)
    if div_yield_pct is not None:
        if div_yield_pct >= 3.0:
            score += 10
        elif div_yield_pct >= 2.0:
            score += 7
        elif div_yield_pct >= 1.0:
            score += 4

    # Payout
    if payout_pct is not None:
        if payout_pct <= 60:
            score += 12
        elif payout_pct <= 80:
            score += 6
        else:
            score -= 8

    # PE
    if pe_ttm is not None:
        if pe_ttm <= 18:
            score += 8
        elif pe_ttm <= 28:
            score += 4
        elif pe_ttm >= 35:
            score -= 6

    # Entry attractiveness
    if to_entry_pct is not None:
        if to_entry_pct >= 5:
            score += 12
        elif to_entry_pct >= 0:
            score += 6
        elif to_entry_pct <= -15:
            score -= 6

    score = max(0, min(100, int(round(score))))
    return score


def load_index_map() -> pd.DataFrame:
    if not INDEX_MAP_PATH.exists():
        return pd.DataFrame(columns=["Ticker", "In SP500", "In Nasdaq100", "In OMXC25"])

    m = pd.read_csv(INDEX_MAP_PATH)
    # require ticker column
    if "ticker" not in [c.lower().strip() for c in m.columns]:
        # try exact
        if "ticker" not in m.columns:
            return pd.DataFrame(columns=["Ticker", "In SP500", "In Nasdaq100", "In OMXC25"])

    # normalize
    colmap = {c.lower().strip(): c for c in m.columns}
    tc = colmap.get("ticker", "ticker")
    sp = colmap.get("sp500")
    nq = colmap.get("nasdaq100")
    omx = colmap.get("omxc25")

    m2 = pd.DataFrame()
    m2["Ticker"] = m[tc].astype(str).str.strip().str.upper()
    m2["In SP500"] = m[sp].fillna(0).astype(int) if sp else 0
    m2["In Nasdaq100"] = m[nq].fillna(0).astype(int) if nq else 0
    m2["In OMXC25"] = m[omx].fillna(0).astype(int) if omx else 0
    return m2


# =========================
# Main
# =========================
def main():
    tickers = read_tickers()
    idx_map = load_index_map()

    rows = []
    for ticker in tickers:
        t = yf.Ticker(ticker)
        info = t.info or {}

        name = info.get("shortName") or info.get("longName") or ""
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        currency = info.get("currency") or ""

        price = get_live_price(t, info)
        market_cap = safe_float(info.get("marketCap"))
        market_cap_bn = round(market_cap / 1e9, 1) if market_cap else None

        # dividend rate (annual)
        div_rate = safe_float(info.get("dividendRate"))
        if div_rate is None:
            div_rate = trailing_12m_dividend(t)

        # payout ratio (often decimal)
        payout = safe_float(info.get("payoutRatio"))
        payout_pct = None if payout is None else (payout * 100.0 if payout <= 1.5 else payout)

        pe_ttm = safe_float(info.get("trailingPE"))

        # compute yield robustly: div_rate / price
        div_yield_pct = None
        if div_rate is not None and price is not None and price > 0:
            div_yield_pct = (div_rate / price) * 100.0

        # target yield: try 5y avg from info (usually already percent), else fall back to current yield
        target_yield_pct_raw = safe_float(info.get("fiveYearAvgDividendYield"))
        target_yield_pct = normalize_target_yield(target_yield_pct_raw, div_yield_pct)

        fair, entry = calc_fair_and_entry(div_rate, target_yield_pct)
        to_fair = pct_diff(fair, price)
        to_entry = pct_diff(entry, price)

        rating = rating_from_entry(price, entry, fair)
        score = quality_score(div_yield_pct, payout_pct, pe_ttm, to_entry)

        country = "DK" if ticker.endswith(".CO") else "US"

        rows.append({
            "Ticker": ticker,
            "Name": name,
            "Country": country,
            "Sector": sector,
            "Industry": industry,
            "Currency": currency,
            "Price": round(price, 2) if price is not None else None,
            "Market Cap (USD bn)": market_cap_bn,
            "Dividend Rate (annual)": round(div_rate, 2) if div_rate is not None else None,
            "Dividend Yield (%)": round(div_yield_pct, 2) if div_yield_pct is not None else None,
            "Payout Ratio (%)": round(payout_pct, 2) if payout_pct is not None else None,
            "PE (TTM)": round(pe_ttm, 2) if pe_ttm is not None else None,
            "Target Yield (%)": round(target_yield_pct, 2) if target_yield_pct is not None else None,
            "Fair Value (Target Yield)": fair,
            "Entry Buy Price": entry,
            "To Fair (%)": to_fair,
            "To Entry (%)": to_entry,
            "Score": score,
            "Rating": rating,
            "Updated (UTC)": now_utc_str(),
        })

    df = pd.DataFrame(rows)

    # merge index flags
    if not idx_map.empty:
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        idx_map["Ticker"] = idx_map["Ticker"].astype(str).str.strip().str.upper()
        df = df.merge(idx_map, on="Ticker", how="left")
        for c in ["In SP500", "In Nasdaq100", "In OMXC25"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = df[c].fillna(0).astype(int)
        df["Index Overlap"] = df[["In SP500", "In Nasdaq100", "In OMXC25"]].sum(axis=1)
    else:
        df["In SP500"] = 0
        df["In Nasdaq100"] = 0
        df["In OMXC25"] = 0
        df["Index Overlap"] = 0

    # sort best first
    rank = {"BUY": 0, "HOLD": 1, "SELL": 2}
    df["_r"] = df["Rating"].map(rank).fillna(9).astype(int)
    df["_te"] = df["To Entry (%)"].fillna(-9999)
    df = df.sort_values(["_r", "Score", "_te"], ascending=[True, False, False]).drop(columns=["_r", "_te"])

    # write docs outputs
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOCS_OUTCSV, index=False)

    # Build HTML with dropdown filters
    table_html = df.to_html(index=False, escape=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Dividend Screener</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; padding: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color:#555; margin: 0 0 18px; }}
    .small {{ font-size: 12px; color:#666; }}
    .tools {{
      display:flex; gap:12px; align-items:center; flex-wrap: wrap;
      margin: 10px 0 18px;
      padding: 12px;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      background: #fafafa;
    }}
    select, input[type="text"] {{
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 10px;
      font-size: 14px;
    }}
    label {{ display:flex; align-items:center; gap:8px; font-size: 14px; color:#222; }}
    .btn {{
      padding:10px 12px; border:1px solid #ccc; border-radius:10px;
      text-decoration:none; color:#111; background:#fff;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    td:first-child, th:first-child {{ text-align: left; }}
    td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
    .pill {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700; border:1px solid #ccc; }}
    .buy {{ background:#e8f7ea; border-color:#9bd3a5; color:#1b6b2a; }}
    .hold {{ background:#fff6dc; border-color:#f0d28b; color:#6a4b00; }}
    .sell {{ background:#fde8e8; border-color:#f3a2a2; color:#7a1111; }}
    .muted {{ color:#777; }}
  </style>
</head>
<body>
  <h1>Dividend Screener</h1>
  <p class="sub">
    Updated automatically via GitHub Actions · <span class="small">{now_utc_str()}</span>
    · <span class="small">CSV: <a href="output.csv">output.csv</a></span>
  </p>

  <div class="tools">
    <input id="searchBox" type="text" placeholder="Search ticker / name / sector..." />

    <select id="ratingFilter">
      <option value="">All ratings</option>
      <option value="BUY">BUY</option>
      <option value="HOLD">HOLD</option>
      <option value="SELL">SELL</option>
    </select>

    <select id="countryFilter">
      <option value="">All countries</option>
      <option value="DK">DK</option>
      <option value="US">US</option>
    </select>

    <select id="sectorFilter">
      <option value="">All sectors</option>
    </select>

    <label>
      <input id="onlyBuy" type="checkbox" />
      Only BUY
    </label>

    <a class="btn" href="output.csv">Download CSV</a>
    <a class="btn" href="#" id="clearBtn">Clear filters</a>
  </div>

  <div class="small muted" id="resultCount"></div>

  {table_html}

  <script>
    const table = document.querySelector("table");
    const headers = Array.from(table.querySelectorAll("th")).map(th => th.innerText.trim());
    const rows = Array.from(table.querySelectorAll("tbody tr"));

    const idx = {{
      ticker: headers.indexOf("Ticker"),
      name: headers.indexOf("Name"),
      sector: headers.indexOf("Sector"),
      rating: headers.indexOf("Rating"),
      country: headers.indexOf("Country"),
    }};

    function getCellText(tr, colIdx) {{
      if (colIdx < 0) return "";
      const td = tr.querySelectorAll("td")[colIdx];
      return (td ? td.innerText : "").trim();
    }}

    // Rating pill styling
    if (idx.rating >= 0) {{
      rows.forEach(tr => {{
        const v = getCellText(tr, idx.rating).toUpperCase();
        const cell = tr.querySelectorAll("td")[idx.rating];
        const cls = v === "BUY" ? "buy" : (v === "HOLD" ? "hold" : "sell");
        cell.innerHTML = `<span class="pill ${cls}">${v}</span>`;
      }});
    }}

    // Populate sector dropdown
    const sectorSet = new Set();
    if (idx.sector >= 0) {{
      rows.forEach(tr => {{
        const v = getCellText(tr, idx.sector);
        if (v) sectorSet.add(v);
      }});
    }}
    const sectorFilter = document.getElementById("sectorFilter");
    Array.from(sectorSet).sort().forEach(v => {{
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      sectorFilter.appendChild(opt);
    }});

    const searchBox = document.getElementById("searchBox");
    const ratingFilter = document.getElementById("ratingFilter");
    const countryFilter = document.getElementById("countryFilter");
    const onlyBuy = document.getElementById("onlyBuy");
    const resultCount = document.getElementById("resultCount");
    const clearBtn = document.getElementById("clearBtn");

    function applyFilters() {{
      const q = (searchBox.value || "").trim().toLowerCase();
      const rating = (onlyBuy.checked ? "BUY" : (ratingFilter.value || "")).toUpperCase();
      const country = (countryFilter.value || "").toUpperCase();
      const sector = (sectorFilter.value || "");

      let shown = 0;

      rows.forEach(tr => {{
        const ticker = getCellText(tr, idx.ticker);
        const name = getCellText(tr, idx.name);
        const sec = getCellText(tr, idx.sector);
        const r = getCellText(tr, idx.rating).toUpperCase();
        const c = getCellText(tr, idx.country).toUpperCase();

        const hay = `${{ticker}} ${{name}} ${{sec}}`.toLowerCase();
        const matchSearch = !q || hay.includes(q);
        const matchRating = !rating || r === rating;
        const matchCountry = !country || c === country;
        const matchSector = !sector || sec === sector;

        const show = matchSearch && matchRating && matchCountry && matchSector;
        tr.style.display = show ? "" : "none";
        if (show) shown += 1;
      }});

      resultCount.textContent = `Showing ${{shown}} of ${{rows.length}}`;
    }}

    [searchBox, ratingFilter, countryFilter, sectorFilter, onlyBuy].forEach(el => {{
      el.addEventListener("input", applyFilters);
      el.addEventListener("change", applyFilters);
    }});

    clearBtn.addEventListener("click", (e) => {{
      e.preventDefault();
      searchBox.value = "";
      ratingFilter.value = "";
      countryFilter.value = "";
      sectorFilter.value = "";
      onlyBuy.checked = false;
      applyFilters();
    }});

    onlyBuy.addEventListener("change", () => {{
      if (onlyBuy.checked) ratingFilter.value = "BUY";
      applyFilters();
    }});

    applyFilters();
  </script>
</body>
</html>
"""

    DOCS_INDEX.write_text(html, encoding="utf-8")
    print(f"✅ Wrote {DOCS_INDEX} and {DOCS_OUTCSV}")


if __name__ == "__main__":
    main()
