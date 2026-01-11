import pandas as pd
import yfinance as yf

tickers = pd.read_csv("input.csv")["ticker"].dropna().astype(str).str.strip().unique()

rows = []

for ticker in tickers:
    t = yf.Ticker(ticker)
    info = t.info or {}

    market_cap = info.get("marketCap")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    div_y = info.get("dividendYield")          # decimal (0.025 = 2.5%)
    payout = info.get("payoutRatio")           # decimal (0.50 = 50%)

    rows.append({
        "Ticker": ticker,
        "Name": info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Currency": info.get("currency"),

        "Price": price,
        "Market Cap (USD bn)": None if market_cap is None else round(market_cap / 1e9, 1),

        "Dividend Yield (%)": None if div_y is None else round(div_y * 100, 2),
        "Dividend Rate (annual)": info.get("dividendRate"),
        "Payout Ratio (%)": None if payout is None else round(payout * 100, 2),

        "PE (TTM)": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "PEG": info.get("pegRatio"),

        "EPS (TTM)": info.get("trailingEps"),
        "EPS Forward": info.get("forwardEps"),

        "Revenue Growth": info.get("revenueGrowth"),
        "Earnings Growth": info.get("earningsGrowth"),

        "Profit Margin": info.get("profitMargins"),
        "Operating Margin": info.get("operatingMargins"),

        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),

        "Debt/Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),

        "Free Cash Flow": info.get("freeCashflow"),
        "Operating Cash Flow": info.get("operatingCashflow"),

        "Beta": info.get("beta"),

        "52w Low": info.get("fiftyTwoWeekLow"),
        "52w High": info.get("fiftyTwoWeekHigh"),
    })

out = pd.DataFrame(rows)
# ---- Scoring + BUY/HOLD/SELL + Entry price ----

# Sæt dine thresholds her
TARGET_YIELD_BUY = 2.5   # % (min yield for at være attraktiv)
MAX_PAYOUT = 70          # % (over dette = risiko)
MAX_PE = 25              # (over dette = dyr)

def to_pct(x):
    return None if x is None else round(x * 100, 2)

def score_and_labels(row):
    score = 0

    y = row.get("Dividend Yield (%)")
    payout = row.get("Payout Ratio (%)")
    pe = row.get("PE (TTM)")
    div_rate = row.get("Dividend Rate (annual)")
    price = row.get("Price")

    # Yield score
    if y is not None:
        if y >= 3.0: score += 35
        elif y >= 2.5: score += 30
        elif y >= 1.5: score += 15
        else: score += 0

    # Payout score
    if payout is not None:
        if payout <= 60: score += 30
        elif payout <= 70: score += 20
        elif payout <= 80: score += 10
        else: score += 0

    # Valuation score
    if pe is not None:
        if pe <= 18: score += 25
        elif pe <= 25: score += 10
        else: score += 0

    # Market cap score (hvis du har den i din out)
    mc = row.get("Market Cap (USD bn)")
    if mc is not None and mc >= 50:
        score += 10

    # Rating
    rating = "SELL"
    if score >= 70:
        rating = "BUY"
    elif score >= 50:
        rating = "HOLD"

    # Entry price baseret på target yield (kun hvis vi har dividend rate)
    entry = None
    upside_to_entry = None
    if div_rate is not None and div_rate != 0:
        entry = round(div_rate / (TARGET_YIELD_BUY / 100), 2)
        if price is not None:
            upside_to_entry = round((entry / price - 1) * 100, 1)

    return score, rating, entry, upside_to_entry

# Sørg for at dine % kolonner er tal i procent (ikke decimal)
# Hvis du i din kode allerede gør: div_y*100 og payout*100, så er det ok.
# Hvis ikke: så skal vi justere – men lad os køre med det du har nu.

out["Score"], out["Rating"], out["Entry Price (Yield Target)"], out["To Entry (%)"] = zip(
    *out.apply(score_and_labels, axis=1)
)

# Sortér så de bedste står øverst
out = out.sort_values(["Rating", "Score"], ascending=[True, False])
print("\nTOP 15 (live preview)")
cols = [c for c in ["Ticker","Name","Dividend Yield (%)","Payout Ratio (%)","PE (TTM)","Score","Rating","Entry Price (Yield Target)","To Entry (%)"] if c in out.columns]
print(out[cols].head(15).to_string(index=False))

out.to_csv("output.csv", index=False)
# ---- Create simple HTML dashboard ----

html_cols = [
    "Ticker",
    "Name",
    "Dividend Yield (%)",
    "Payout Ratio (%)",
    "PE (TTM)",
    "Score",
    "Rating",
    "Entry Price (Yield Target)",
    "To Entry (%)"
]

html_df = out[html_cols].copy()

html = f"""
<html>
<head>
<title>Dividend Screener</title>
<style>
body {{ font-family: Arial; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: right; }}
th {{ background: #eee; }}
td:first-child, th:first-child {{ text-align: left; }}
.buy {{ background: #c6efce; }}
.hold {{ background: #fff2cc; }}
.sell {{ background: #f4cccc; }}
</style>
</head>
<body>

<h1>Dividend Screener</h1>
<p>Updated automatically via GitHub Actions</p>

{html_df.to_html(index=False, classes="table")}
</body>
</html>
"""

from pathlib import Path
Path("docs").mkdir(exist_ok=True)
Path("docs/index.html").write_text(html, encoding="utf-8")

print("HTML dashboard written to docs/index.html")


print(f"Saved output.csv with {len(out)} rows and {len(out.columns)} columns")
