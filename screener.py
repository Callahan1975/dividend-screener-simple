#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener (DK + US) – Portfolio-aware dashboard

Fixes:
- Percent fields (DividendYield, DividendGrowth5Y, Upside_*) are normalized so you never get 300%+ yields by mistake.
- Generates docs/index.html with built-in filters (DataTables + dropdown filters).
- Integrates Snowball portfolio CSV + rules => PortfolioAction = ADD/BUY/HOLD/TRIM/AVOID.

Files expected:
- tickers.txt
- data/portfolio/Snowball.csv
- data/portfolio/ticker_alias.csv   (alias,ticker)
- config/portfolio_rules.yml
Outputs:
- data/screener_results.csv
- docs/index.html
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf
import yaml

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parent

TICKERS_TXT = ROOT / "tickers.txt"
OUT_CSV = ROOT / "data" / "screener_results.csv"
OUT_HTML = ROOT / "docs" / "index.html"

SNOWBALL_PATH = ROOT / "data" / "portfolio" / "Snowball.csv"
ALIAS_PATH = ROOT / "data" / "portfolio" / "ticker_alias.csv"
RULES_PATH = ROOT / "config" / "portfolio_rules.yml"


# -------------------------
# Helpers
# -------------------------
def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            if np.isnan(x):
                return None
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def normalize_pct(value):
    """
    Normalize percent-like inputs into "percent points" (e.g., 3.4 means 3.4%).
    Rules:
      - None -> None
      - if abs(value) <= 1.5 => treat as fraction (0.034) and multiply by 100
      - else => already percent points (3.4) keep as-is
    """
    v = safe_float(value)
    if v is None:
        return None
    if abs(v) <= 1.5:
        return v * 100.0
    return v


def fmt_pct(v, digits=1):
    v = safe_float(v)
    if v is None:
        return ""
    return f"{v:.{digits}f}%"


def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line)
    return tickers


def yahoo_price(info: dict) -> float | None:
    # yfinance info keys vary; try common ones
    for k in ["currentPrice", "regularMarketPrice", "previousClose"]:
        if k in info and info[k] is not None:
            return safe_float(info[k])
    return None


def get_company_name(info: dict, fallback: str) -> str:
    for k in ["shortName", "longName"]:
        if k in info and info[k]:
            return str(info[k])
    return fallback


def get_payout_ratio(info: dict) -> float | None:
    # payout ratio is usually a fraction (0.45) or already percent-ish depending on source
    v = info.get("payoutRatio", None)
    return safe_float(v)


def get_pe(info: dict) -> float | None:
    for k in ["trailingPE", "forwardPE"]:
        v = safe_float(info.get(k))
        if v is not None and v > 0:
            return v
    return None


def dividend_growth_5y(ticker: yf.Ticker) -> float | None:
    """
    Approx 5Y dividend growth from annual sums.
    Returns fraction (0.10 = 10%) internally; we normalize later.
    """
    try:
        div = ticker.dividends
        if div is None or len(div) < 8:
            return None
        # yearly totals
        yearly = div.resample("Y").sum()
        if len(yearly) < 6:
            return None
        # last 6 years gives 5 intervals
        last = yearly.iloc[-1]
        first = yearly.iloc[-6]
        first = float(first) if first is not None else 0.0
        last = float(last) if last is not None else 0.0
        if first <= 0 or last <= 0:
            return None
        cagr = (last / first) ** (1 / 5) - 1
        if not np.isfinite(cagr):
            return None
        return float(cagr)
    except Exception:
        return None


def fair_value_yield(price: float | None, yield_for_model_pct: float | None) -> float | None:
    """
    Very simple fair value from yield.
    If yield_for_model is percent points (e.g., 3.0), then FV = annual_div / (yield_for_model/100).
    We approximate annual_div = price * (current_yield/100) if we have yield_for_model same as current (fallback).
    """
    if price is None or price <= 0:
        return None
    y = safe_float(yield_for_model_pct)
    if y is None or y <= 0:
        return None
    # This is "price" level that would correspond to that yield if dividends unchanged.
    # So fair value from yield is: FV = price * (current_yield / y_model).
    # We'll do this outside when we have both yields.
    return None


def upside_pct(price: float | None, fair: float | None) -> float | None:
    if price is None or fair is None or price <= 0:
        return None
    return (fair / price - 1.0)


# -------------------------
# Portfolio ingest + actions
# -------------------------
def load_alias_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    # expected headers: alias,ticker
    cols = {c.lower(): c for c in df.columns}
    if "alias" not in cols or "ticker" not in cols:
        return {}
    a = df[cols["alias"]].astype(str).str.strip()
    t = df[cols["ticker"]].astype(str).str.strip()
    out = {}
    for aa, tt in zip(a, t):
        if aa and tt and aa.lower() != "nan" and tt.lower() != "nan":
            out[aa] = tt
    return out


def build_positions_from_snowball(path: Path, alias_map: dict[str, str]) -> pd.DataFrame:
    """
    Snowball exports can differ. We handle the common "Holdings" style where there is a Symbol + Shares or Quantity.
    If your Snowball has different headers, we’ll still keep this tolerant by searching for likely columns.
    """
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "Shares"])

    df = pd.read_csv(path)

    # Find symbol column
    sym_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"symbol", "ticker", "instrument", "name"}:
            sym_col = c
            break
    if sym_col is None:
        # fallback: if there is a 'Holding' column with symbols
        for c in df.columns:
            if "symbol" in c.lower() or "ticker" in c.lower():
                sym_col = c
                break
    if sym_col is None:
        # cannot parse
        return pd.DataFrame(columns=["Ticker", "Shares"])

    # Find shares/quantity
    qty_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"shares", "quantity", "antal", "units"}:
            qty_col = c
            break
        if "share" in cl and qty_col is None:
            qty_col = c
    if qty_col is None:
        return pd.DataFrame(columns=["Ticker", "Shares"])

    pos = df[[sym_col, qty_col]].copy()
    pos.columns = ["Ticker", "Shares"]
    pos["Ticker"] = pos["Ticker"].astype(str).str.strip()
    pos["Shares"] = pd.to_numeric(pos["Shares"], errors="coerce").fillna(0.0)

    # apply alias mapping to Yahoo tickers
    pos["Ticker"] = pos["Ticker"].map(lambda x: alias_map.get(x, x))

    # aggregate
    pos = pos.groupby("Ticker", as_index=False)["Shares"].sum()
    pos = pos[pos["Shares"].abs() > 1e-9]
    return pos


def apply_portfolio_context(df: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["OwnedShares"] = 0.0
    out["OwnedValue"] = 0.0
    out["Weight"] = 0.0

    if positions is None or positions.empty:
        return out

    pos_map = dict(zip(positions["Ticker"], positions["Shares"]))
    out["OwnedShares"] = out["Ticker"].map(lambda t: float(pos_map.get(t, 0.0)))
    out["OwnedValue"] = out["OwnedShares"] * pd.to_numeric(out["Price"], errors="coerce").fillna(0.0)

    total = float(out["OwnedValue"].sum())
    if total > 0:
        out["Weight"] = out["OwnedValue"] / total
    else:
        out["Weight"] = 0.0

    return out


def decide_portfolio_actions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Produces PortfolioAction using simple rules:
      - If not owned: BUY if Reco is Strong/Ok and upside positive; else AVOID
      - If owned: HOLD by default
      - TRIM if weight > max_weight or payout ratio too high
      - ADD if owned but under min_weight and still attractive
    """
    out = df.copy()

    # Defaults
    max_weight = float(rules.get("max_weight", 0.08))  # 8%
    min_add_weight = float(rules.get("min_add_weight", 0.02))  # 2%
    trim_weight = float(rules.get("trim_weight", max_weight))
    payout_trim = float(rules.get("payout_ratio_trim", 0.85))  # 85% payout ratio (fraction)

    def classify(row):
        owned = float(row.get("OwnedShares", 0.0)) > 0
        w = safe_float(row.get("Weight")) or 0.0
        reco = str(row.get("Reco") or "").strip().upper()
        upside = safe_float(row.get("Upside_Yield_%"))  # percent points
        payout = safe_float(row.get("PayoutRatio"))
        # payout might be fraction; if it's > 1.5 we interpret as percent points and convert
        if payout is not None and payout > 1.5:
            payout = payout / 100.0

        attractive = (reco in {"STRONG", "OK"}) and (upside is None or upside >= 0)

        if not owned:
            return "BUY" if attractive else "AVOID"

        # owned logic:
        if w >= trim_weight:
            return "TRIM"
        if payout is not None and payout >= payout_trim:
            return "TRIM"
        if attractive and w < min_add_weight:
            return "ADD"
        return "HOLD"

    out["PortfolioAction"] = out.apply(classify, axis=1)
    return out


# -------------------------
# Core screener
# -------------------------
def build_screener_df(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            info = {}

        price = yahoo_price(info)
        name = get_company_name(info, t)

        dy = info.get("dividendYield", None)  # usually fraction
        dy_pct = normalize_pct(dy)

        dgr = dividend_growth_5y(tk)  # fraction
        dgr_pct = normalize_pct(dgr)

        payout = get_payout_ratio(info)  # fraction (usually)
        pe = get_pe(info)

        # Simple "yield model" = current yield (percent points), used for fair value yield
        yield_for_model = dy_pct

        # fair value from yield: FV = price * (current_yield / model_yield) -> here equals price (so not useful)
        # Instead we use a conservative "target yield" = max(current_yield, 3.0) just as example.
        # You can tweak later in rules. This keeps FV stable and avoids insane upside.
        target_yield = max((dy_pct or 0.0), 3.0) if dy_pct is not None else 3.0
        # approximate FV_yield: price * (dy_pct / target_yield)
        fair_yield = None
        if price is not None and price > 0 and dy_pct is not None and target_yield > 0:
            fair_yield = price * (dy_pct / target_yield)

        upside_yield = normalize_pct(upside_pct(price, fair_yield))  # careful: upside_pct returns fraction
        # upside_pct returns fraction, normalize_pct will convert to percent points (x100)

        rows.append(
            {
                "Ticker": t,
                "Name": name,
                "Price": price,
                "DividendYield": dy_pct,          # percent points
                "DividendGrowth5Y": dgr_pct,      # percent points
                "PayoutRatio": payout,            # fraction
                "PE": pe,
                "FairValue_Yield": fair_yield,
                "Upside_Yield_%": upside_yield,   # percent points
            }
        )

    df = pd.DataFrame(rows)

    # score + reco
    def score_row(r):
        y = safe_float(r["DividendYield"]) or 0.0
        g = safe_float(r["DividendGrowth5Y"]) or 0.0
        pe = safe_float(r["PE"])
        pe_pen = 0.0 if (pe is None or pe <= 0) else min(20.0, max(0.0, (pe - 15.0) * 0.5))
        return max(0.0, min(100.0, (y * 8.0 + g * 2.0) - pe_pen))

    df["Score"] = df.apply(score_row, axis=1)

    def reco_label(score):
        s = safe_float(score) or 0.0
        if s >= 70:
            return "Strong"
        if s >= 50:
            return "Ok"
        if s >= 35:
            return "Weak"
        return "Avoid"

    df["Reco"] = df["Score"].map(reco_label)
    df["Action"] = df["Reco"].map(lambda r: "BUY" if r in {"Strong", "Ok"} else "HOLD")

    # Pretty display columns (string)
    df["DividendYield"] = df["DividendYield"].map(lambda v: fmt_pct(v, 1))
    df["DividendGrowth5Y"] = df["DividendGrowth5Y"].map(lambda v: fmt_pct(v, 1))
    df["Upside_Yield_%"] = df["Upside_Yield_%"].map(lambda v: fmt_pct(v, 1))
    df["PayoutRatio"] = df["PayoutRatio"].map(lambda v: "" if safe_float(v) is None else f"{safe_float(v):.2f}")
    df["PE"] = df["PE"].map(lambda v: "" if safe_float(v) is None else f"{safe_float(v):.2f}")
    df["FairValue_Yield"] = df["FairValue_Yield"].map(lambda v: "" if safe_float(v) is None else f"{safe_float(v):.2f}")
    df["Price"] = df["Price"].map(lambda v: "" if safe_float(v) is None else f"{safe_float(v):.2f}")

    return df


# -------------------------
# Dashboard HTML (with filters)
# -------------------------
def write_dashboard_html(df: pd.DataFrame, out_html: Path):
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # DataTables + dropdown filters for key columns
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Dividend Screener (DK + US)</title>

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/datatables.net-dt@2.1.8/css/dataTables.dataTables.min.css">
  <style>
    body {{
      font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 24px;
    }}
    h1 {{ margin: 0 0 6px 0; }}
    .meta {{ color: #666; margin-bottom: 16px; }}
    .filters {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 12px 0 16px 0;
      align-items: center;
    }}
    .filters label {{
      font-size: 12px;
      color: #333;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    select {{
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #ddd;
      min-width: 180px;
    }}
    table.dataTable tbody td {{ vertical-align: top; }}
  </style>
</head>
<body>
  <h1>Dividend Screener (DK + US)</h1>
  <div class="meta">Updated: {updated}</div>

  <div class="filters">
    <label>Reco
      <select id="fReco"><option value="">All</option></select>
    </label>
    <label>Action
      <select id="fAction"><option value="">All</option></select>
    </label>
    <label>PortfolioAction
      <select id="fPort"><option value="">All</option></select>
    </label>
  </div>

  {df.to_html(index=False, escape=False, table_id="tbl", classes="display", border=0)}

  <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/datatables.net@2.1.8/js/dataTables.min.js"></script>
  <script>
    function fillSelectFromColumn(table, colIdx, selectId) {{
      const api = table;
      const data = new Set();
      api.column(colIdx).data().each(function(v) {{
        if (v !== null && v !== undefined && String(v).trim() !== "") data.add(String(v).trim());
      }});
      const sel = document.getElementById(selectId);
      Array.from(data).sort().forEach(v => {{
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      }});
    }}

    $(document).ready(function() {{
      const table = $('#tbl').DataTable({{
        pageLength: 50,
        order: []
      }});

      // Find column indexes by header text
      function colIndex(name) {{
        let idx = -1;
        $('#tbl thead th').each(function(i) {{
          if ($(this).text().trim() === name) idx = i;
        }});
        return idx;
      }}

      const recoCol = colIndex("Reco");
      const actionCol = colIndex("Action");
      const portCol = colIndex("PortfolioAction");

      if (recoCol >= 0) fillSelectFromColumn(table, recoCol, "fReco");
      if (actionCol >= 0) fillSelectFromColumn(table, actionCol, "fAction");
      if (portCol >= 0) fillSelectFromColumn(table, portCol, "fPort");

      function applySelectFilter(selectId, colIdx) {{
        document.getElementById(selectId).addEventListener('change', function() {{
          const val = this.value;
          if (!val) {{
            table.column(colIdx).search('').draw();
          }} else {{
            table.column(colIdx).search('^' + val.replace(/[.*+?^{{}}()|[\\]\\\\]/g, '\\\\$&') + '$', true, false).draw();
          }}
        }});
      }}

      if (recoCol >= 0) applySelectFilter("fReco", recoCol);
      if (actionCol >= 0) applySelectFilter("fAction", actionCol);
      if (portCol >= 0) applySelectFilter("fPort", portCol);
    }});
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    tickers = read_tickers(TICKERS_TXT)
    df = build_screener_df(tickers)

    # Portfolio integration
    alias_map = load_alias_map(ALIAS_PATH)
    positions = build_positions_from_snowball(SNOWBALL_PATH, alias_map)

    rules = {}
    if RULES_PATH.exists():
        rules = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}

    df = apply_portfolio_context(df, positions)
    df = decide_portfolio_actions(df, rules)

    # Ensure column order with portfolio columns last-ish
    preferred = [
        "Ticker","Name","Price","DividendYield","DividendGrowth5Y","PayoutRatio","PE",
        "FairValue_Yield","Upside_Yield_%","Score","Reco","Action",
        "OwnedShares","OwnedValue","Weight","PortfolioAction"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    write_dashboard_html(df, OUT_HTML)

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_HTML}")


if __name__ == "__main__":
    main()
