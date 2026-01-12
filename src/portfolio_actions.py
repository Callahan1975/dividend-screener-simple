from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import yaml


DEFAULT_RULES = {
    # portfolio sizing
    "max_position_weight": 0.06,   # 6% max per position
    "min_position_weight": 0.01,   # 1% min "starter"
    "trim_over": 1.20,             # trim if weight > 1.2 * max
    "add_under": 0.80,             # add if weight < 0.8 * min (and you like it)
    # thresholds using screener metrics
    "buy_score_min": 70.0,
    "add_score_min": 60.0,
    "avoid_score_max": 45.0,
    "buy_upside_min": 0.07,
    "add_upside_min": 0.03,
    # optional: always hold these tickers (never AVOID)
    "always_hold": [],
}


def load_rules(rules_path: Path) -> dict:
    if rules_path is None or not Path(rules_path).exists():
        return dict(DEFAULT_RULES)
    try:
        d = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
        out = dict(DEFAULT_RULES)
        out.update(d)
        # normalize list
        out["always_hold"] = list(out.get("always_hold") or [])
        return out
    except Exception:
        return dict(DEFAULT_RULES)


def _safe_series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df), index=df.index)


def apply_portfolio_actions(
    screener_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Adds:
      OwnedShares, OwnedValue, Weight, PortfolioAction
    Keeps:
      Action as baseline screener action
    """
    out = screener_df.copy()

    # Ensure baseline Action exists
    if "Action" not in out.columns:
        out["Action"] = "HOLD"

    # normalize tickers
    out["Ticker"] = out["Ticker"].astype(str).str.strip()

    # positions
    pos = positions_df.copy() if positions_df is not None else pd.DataFrame(columns=["Ticker", "Shares"])
    if not pos.empty:
        pos["Ticker"] = pos["Ticker"].astype(str).str.strip()
        pos["Shares"] = pd.to_numeric(pos["Shares"], errors="coerce").fillna(0.0)
        pos = pos[pos["Shares"].abs() > 1e-9].copy()

    # merge shares
    out = out.merge(pos[["Ticker", "Shares"]], on="Ticker", how="left")
    out.rename(columns={"Shares": "OwnedShares"}, inplace=True)
    out["OwnedShares"] = pd.to_numeric(out["OwnedShares"], errors="coerce").fillna(0.0)

    # owned value uses price
    price = _safe_series_numeric(out, "Price")
    out["OwnedValue"] = out["OwnedShares"] * price

    total = float(out["OwnedValue"].sum())
    if total <= 0:
        out["Weight"] = 0.0
    else:
        out["Weight"] = out["OwnedValue"] / total

    # numeric metrics (safe even if missing)
    score = _safe_series_numeric(out, "Score")
    upside = _safe_series_numeric(out, "Upside_Yield_%")

    # rules
    maxw = float(rules.get("max_position_weight", DEFAULT_RULES["max_position_weight"]))
    minw = float(rules.get("min_position_weight", DEFAULT_RULES["min_position_weight"]))
    trim_over = float(rules.get("trim_over", DEFAULT_RULES["trim_over"]))
    add_under = float(rules.get("add_under", DEFAULT_RULES["add_under"]))
    buy_score_min = float(rules.get("buy_score_min", DEFAULT_RULES["buy_score_min"]))
    add_score_min = float(rules.get("add_score_min", DEFAULT_RULES["add_score_min"]))
    avoid_score_max = float(rules.get("avoid_score_max", DEFAULT_RULES["avoid_score_max"]))
    buy_upside_min = float(rules.get("buy_upside_min", DEFAULT_RULES["buy_upside_min"]))
    add_upside_min = float(rules.get("add_upside_min", DEFAULT_RULES["add_upside_min"]))
    always_hold = set([str(x).strip() for x in (rules.get("always_hold") or [])])

    weight = out["Weight"].astype(float)

    def decide(row) -> str:
        tk = str(row["Ticker"]).strip()
        owned = float(row["OwnedShares"]) > 0
        w = float(row["Weight"]) if total > 0 else 0.0
        sc = float(row["Score"]) if not pd.isna(row["Score"]) else 0.0
        up = float(row.get("Upside_Yield_%") or 0.0)

        # never force AVOID on these
        if tk in always_hold:
            if owned and w > maxw * trim_over:
                return "TRIM"
            return "HOLD"

        if owned:
            if w > maxw * trim_over:
                return "TRIM"
            # add if position is too small AND fundamentals are good
            if w < minw * add_under and sc >= add_score_min and up >= add_upside_min:
                return "ADD"
            # avoid if very weak
            if sc <= avoid_score_max and up <= 0:
                return "TRIM"
            return "HOLD"

        # not owned:
        if sc >= buy_score_min and up >= buy_upside_min:
            return "BUY"
        if sc >= add_score_min and up >= add_upside_min:
            return "ADD"
        if sc <= avoid_score_max:
            return "AVOID"
        return "HOLD"

    out["PortfolioAction"] = out.apply(decide, axis=1)
    return out
