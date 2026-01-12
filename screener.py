import os
from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml

from src.portfolio_ingest import build_positions
from src.portfolio_actions import apply_portfolio_context, decide_actions

# ---------------- CONFIG ----------------
TICKERS_FILE = "tickers.txt"
SNOWBALL_PATH = "data/portfolio/Snowball.csv"
RULES_PATH = "config/portfolio_rules.yml"
ALIAS_PATH = "data/portfolio/ticker_alias.csv"

OUT_CSV = Path("data/screener_results/screener_results.csv")
# ---------------------------------------


def read_tickers(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0])
    return sorted(set(out))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_ratio(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x / 100 if abs(x) > 1 else x


def load_alias_map():
    if not os.path.exists(ALIAS_PATH):
        return {}
    try:
        df = pd.read_csv(ALIAS_PATH)
        df.columns = [c.strip() for c in df.columns]
        if len(df.columns) < 2:
            return {}
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    except Exception:
        return {}


def main():
    # 1) Load tickers
    tickers = read_tickers(TICKERS_FILE)

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}

        rows.append({
            "Ticker": t,
            "Price": safe_float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            ),
            "DividendYield": safe_float(info.get("dividendYield")),
            "PE": safe_float(info.get("trailingPE"), None),
        })

    df = pd.DataFrame(rows)
    df["YieldPct"] = (df["DividendYield"] * 100).round(2)

    # Simple upside proxy (kan raffineres senere)
    df["UpsidePct"] = df["YieldPct"].apply(lambda y: 0.15 if y >= 3 else 0.05)

    # Required columns for portfolio engine
    df["DividendYears"] = 20
    df["FCFPositive"] = True

    # 2) Portfolio integration
    if os.path.exists(SNOWBALL_PATH) and os.path.exists(RULES_PATH):
        pos_df = build_positions(SNOWBALL_PATH)

        alias_map = load_alias_map()
        if alias_map:
            pos_df["Symbol"] = pos_df["Symbol"].map(lambda s: alias_map.get(s, s))

        with open(RULES_PATH, "r") as f:
            rules = yaml.safe_load(f) or {}

        df = apply_portfolio_context(df, pos_df)
        df = decide_actions(df, rules)

        df["PortfolioAction"] = df["Action"]
    else:
        df["OwnedShares"] = 0
        df["OwnedValue"] = 0
        df["Weight"] = 0
        df["PortfolioAction"] = "HOLD"
        df["Action"] = "HOLD"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
