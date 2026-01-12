import pandas as pd

def apply_portfolio_context(screen_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    out = screen_df.copy()

    pos = pos_df.rename(columns={"Symbol": "Ticker", "Shares": "OwnedShares"}).copy()
    pos["Ticker"] = pos["Ticker"].astype(str).str.strip()

    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out = out.merge(pos, on="Ticker", how="left")
    out["OwnedShares"] = out["OwnedShares"].fillna(0.0)

    out["OwnedValue"] = out["OwnedShares"] * pd.to_numeric(out["Price"], errors="coerce").fillna(0.0)
    total_value = out["OwnedValue"].sum()
    out["Weight"] = (out["OwnedValue"] / total_value) if total_value > 0 else 0.0

    return out

def decide_actions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    out = df.copy()

    max_w = float(rules.get("max_weight", 0.08))
    min_w = float(rules.get("min_weight", 0.02))

    q = rules.get("quality", {})
    v = rules.get("valuation", {})

    min_years = int(q.get("min_dividend_years", 5))
    max_payout = float(q.get("max_payout_ratio", 0.80))
    require_fcf = bool(q.get("require_positive_fcf", True))

    buy_upside = float(v.get("buy_fair_value_upside_min", 0.10))
    trim_overval = float(v.get("trim_overvaluation_min", 0.20))

    out["UpsidePct"] = pd.to_numeric(out.get("UpsidePct"), errors="coerce").fillna(0.0)
    if "PayoutRatio" in out.columns:
    out["PayoutRatio"] = pd.to_numeric(out["PayoutRatio"], errors="coerce").fillna(0.0)
else:
    out["PayoutRatio"] = 0.0

    out["DividendYears"] = pd.to_numeric(out.get("DividendYears"), errors="coerce").fillna(0.0)
    out["FCFPositive"] = out.get("FCFPositive", True)

    held = out["OwnedShares"] > 0
    overweight = out["Weight"] > max_w
    underweight = (out["Weight"] < min_w) & held

    quality_fail = (out["DividendYears"] < min_years) | (out["PayoutRatio"] > max_payout)
    if require_fcf:
        quality_fail = quality_fail | (~out["FCFPositive"].astype(bool))

    attractive = out["UpsidePct"] >= buy_upside
    very_overvalued = out["UpsidePct"] <= -trim_overval

    out["Action"] = "AVOID"
    out.loc[~held & ~quality_fail & attractive, "Action"] = "BUY"
    out.loc[~held & ~quality_fail & ~attractive, "Action"] = "HOLD"

    out.loc[held & quality_fail, "Action"] = "AVOID"
    out.loc[held & ~quality_fail & underweight & attractive, "Action"] = "ADD"
    out.loc[held & ~quality_fail & ~overweight, "Action"] = "HOLD"
    out.loc[held & ~quality_fail & overweight, "Action"] = "TRIM"
    out.loc[held & ~quality_fail & overweight & very_overvalued, "Action"] = "TRIM"

    return out
