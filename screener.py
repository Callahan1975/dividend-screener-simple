import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

DATA_FILE = "tickers.txt"
OUT_DIR = Path("docs")
OUT_FILE = OUT_DIR / "index.html"

# ----------------------------
# Helpers
# ----------------------------
def safe(v):
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def pct(v, d=1):
    return "" if v is None else f"{v:.{d}f}%"


def num(v, d=2):
    return "" if v is None else f"{v:.{d}f}"


# ----------------------------
# Load tickers
# ----------------------------
def load_tickers():
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError("tickers.txt mangler")
    with open(DATA_FILE) as f:
        return [l.strip() for l in f if l.strip()]


# ----------------------------
# Fetch data
# ----------------------------
def fetch(ticker):
    t = yf.Ticker(ticker)
    info = t.info

    price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    dps = safe(info.get("dividendRate"))
    yield_pct = None
    if price and dps:
        yield_pct = dps / price * 100

    pe = safe(info.get("trailingPE"))
    growth5y = safe(info.get("earningsGrowth"))
    if growth5y:
        growth5y *= 100

    # Fair value (yield model, conservative)
    fair_yield = None
    if dps:
        fair_yield = dps / 0.03  # 3% required yield

    # Fair value (simple DDM)
    fair_ddm = None
    if dps and growth5y:
        g = min(growth5y / 100, 0.06)
        r = 0.09
        if r > g:
            fair_ddm = dps * (1 + g) / (r - g)

    upside = None
    if fair_yield and price:
        upside = (fair_yield / price - 1) * 100

    score = 0
    if yield_pct:
        score += min(yield_pct * 10, 40)
    if growth5y:
        score += min(growth5y, 30)
    if pe and pe < 20:
        score += 20

    reco = "HOLD"
    if upside and upside > 15 and score > 70:
        reco = "BUY"
    if upside is not None and upside < 5:
        reco = "WATCH"

    return {
        "Ticker": ticker,
        "Name": info.get("shortName"),
        "Price": price,
        "PE": pe,
        "Yield": yield_pct,
        "DivG5Y": growth5y,
        "FairYield": fair_yield,
        "FairDDM": fair_ddm,
        "Upside": upside,
        "Score": score,
        "Reco": reco,
        "Sector": info.get("sector"),
        "Country": info.get("country"),
    }


# ----------------------------
# Build HTML (NO f-strings!)
# ----------------------------
def build_html(rows):
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    head = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dividend Screener</title>
<style>
body{background:#0b0f14;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto}
table{width:100%;border-collapse:collapse}
th,td{padding:8px;border-bottom:1px solid #223042;font-size:13px}
th{color:#9aa4af;text-align:left}
.right{text-align:right}
.pill{padding:3px 10px;border-radius:999px;font-size:12px}
.buy{background:rgba(26,173,89,.2)}
.hold{background:rgba(255,200,64,.2)}
.watch{background:rgba(148,163,184,.15)}
</style>
</head>
<body>
<h2>Dividend Screener (DK + US)</h2>
<div style="color:#9aa4af;font-size:12px">Generated: {gen}</div>
<table>
<thead>
<tr>
<th>Ticker</th><th>Name</th><th class="right">Price</th>
<th class="right">PE</th><th class="right">Yield</th>
<th class="right">Div G 5Y</th>
<th class="right">Fair (Yield)</th>
<th class="right">Fair (DDM)</th>
<th class="right">Upside</th>
<th class="right">Score</th>
<th>Reco</th>
<th>Sector</th><th>Country</th>
</tr>
</thead>
<tbody>
""".format(gen=gen)

    body = ""
    for r in rows:
        pill = "hold"
        if r["Reco"] == "BUY":
            pill = "buy"
        if r["Reco"] == "WATCH":
            pill = "watch"

        body += """
<tr>
<td>{Ticker}</td>
<td>{Name}</td>
<td class="right">{Price}</td>
<td class="right">{PE}</td>
<td class="right">{Yield}</td>
<td class="right">{DivG5Y}</td>
<td class="right">{FairYield}</td>
<td class="right">{FairDDM}</td>
<td class="right">{Upside}</td>
<td class="right">{Score}</td>
<td><span class="pill {pill}">{Reco}</span></td>
<td>{Sector}</td>
<td>{Country}</td>
</tr>
""".format(
            Ticker=r["Ticker"],
            Name=r["Name"] or "",
            Price=num(r["Price"]),
            PE=num(r["PE"], 1),
            Yield=pct(r["Yield"]),
            DivG5Y=pct(r["DivG5Y"]),
            FairYield=num(r["FairYield"]),
            FairDDM=num(r["FairDDM"]),
            Upside=pct(r["Upside"]),
            Score=num(r["Score"], 1),
            Reco=r["Reco"],
            Sector=r["Sector"] or "",
            Country=r["Country"] or "",
            pill=pill,
        )

    tail = """
</tbody>
</table>
</body>
</html>
"""

    return head + body + tail


# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for t in load_tickers():
        try:
            rows.append(fetch(t))
        except Exception as e:
            print("Skip", t, e)

    df = pd.DataFrame(rows)
    df = df.sort_values("Upside", ascending=False)

    html = build_html(df.to_dict("records"))
    OUT_FILE.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
