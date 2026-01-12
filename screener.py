"""
Portfolio-aware screener wrapper.

- Imports the existing screener logic from screener.py
- Runs it to produce df
- Adds Snowball portfolio context + ADD/BUY/HOLD/TRIM/AVOID using config/portfolio_rules.yml
- Writes a separate output CSV so the original workflow remains safe.
"""

import os
import pandas as pd
import yaml

from src.portfolio_ingest import build_positions
from src.portfolio_actions import apply_portfolio_context, decide_actions

# --- Paths ---
SNOWBALL_PATH = "data/portfolio/Snowball.csv"   # your file name (capital S)
RULES_PATH = "config/portfolio_rules.yml"
ALIAS_PATH = "data/portfolio/ticker_alias.csv"

# --- Output (separate file, safer) ---
OUT_CSV_PORTFOLIO = "data/screener_results/screener_results_portfolio.csv"

def _to_ratio(x):
    """Convert percent-like numbers to ratio. 12.3 -> 0.123; 0.12 stays 0.12"""
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x / 100.0 if abs(x) > 1.0 else x

def _apply_aliases(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Map Snowball symbols to the screener tickers using ticker_alias.csv (first 2 columns)."""
    if not os.path.exists(ALIAS_PATH):
        return pos_df

    try:
        alias_df = pd.read_csv(ALIAS_PATH)
        alias_df.columns = [c.strip() for c in alias_df.columns]
        cols = list(alias_df.columns)
        if len(cols) < 2:
            return pos_df

        alias_col, ticker_col = cols[0], cols[1]
        mapping = dict(
            zip(
                alias_df[alias_col].astype(str).str.strip(),
                alias_df[ticker_col].astype(str).str.strip(),
            )
        )
        out = pos_df.copy()
        out["Symbol"] = out["Symbol"].astype(str).str.strip().map(lambda s: mapping.get(s, s))
        return out
    except Exception:
        return pos_df

def _add_upsidepct(df: pd.DataFrame) -> pd.DataFrame:
    """Create UpsidePct ratio column (0.10=10%) from existing Upside columns."""
    out = df.copy()

    if "Ticker" not in out.columns and "Symbol" in out.columns:
        out["Ticker"] = out["Symbol"]

    if "Upside_Yield_%" in out.columns and "Upside_DDM_%" in out.columns:
        out["UpsidePct"] = out[["Upside_Yield_%", "Upside_DDM_%"]].max(axis=1).apply(_to_ratio)
    elif "Upside_Yield_%" in out.columns:
        out["UpsidePct"] = out["Upside_Yield_%"].apply(_to_ratio)
    elif "Upside_DDM_%" in out.columns:
        out["UpsidePct"] = out["Upside_DDM_%"].apply(_to_ratio)
    else:
        out["UpsidePct"] = 0.0

    return out

def add_portfolio_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach OwnedShares/Weight and compute PortfolioAction."""
    out = df.copy()

    # keep original action columns (from your current screener)
    if "Action" in out.columns:
        out["ScreenerAction"] = out["Action"]

    out = _add_upsidepct(out)

    # if missing required files, just return df with safe defaults
    if not (os.path.exists(SNOWBALL_PATH) and os.path.exists(RULES_PATH)):
        out["OwnedShares"] = 0
        out["OwnedValue"] = 0
        out["Weight"] = 0
        out["PortfolioAction"] = "HOLD"
        return out

    pos_df = build_positions(SNOWBALL_PATH)
    pos_df = _apply_aliases(pos_df)

    with open(RULES_PATH, "r") as f:
        rules = yaml.safe_load(f) or {}

    out = apply_portfolio_context(out, pos_df)
    out = decide_actions(out, rules)

    # decide_actions writes into "Action"
    out["PortfolioAction"] = out["Action"]

    # keep both; final Action becomes portfolio-driven
    if "ScreenerAction" in out.columns:
        out["Action"] = out["PortfolioAction"]

    return out

def main():
    # Import your existing screener and run it to get df.
    # Your screener.py already builds df and writes output when executed as a script.
    # We need a function to call; easiest: import and call a "run()" if it exists.
    # If it doesn't exist, we fallback to running screener.py as a module is tricky.
    #
    # Therefore: we expect screener.py exposes a function `run_screener()` returning df.
    # If not, follow instructions below to add that small function (safe, minimal).
    from screener import run_screener  # type: ignore

    df = run_screener()
    df2 = add_portfolio_actions(df)

    os.makedirs(os.path.dirname(OUT_CSV_PORTFOLIO), exist_ok=True)
    df2.to_csv(OUT_CSV_PORTFOLIO, index=False, encoding="utf-8")

    print(f"Wrote: {OUT_CSV_PORTFOLIO} ({len(df2)} rows)")

if __name__ == "__main__":
    main()
