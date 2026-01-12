# src/portfolio_actions.py
# Portfolio integration + action engine (ADD / BUY / HOLD / TRIM / AVOID)
#
# This file is intentionally defensive:
# - Works even if some columns are missing
# - Never calls .fillna() on scalars
# - Handles both "Ticker" and "Symbol" naming
# - Uses Snowball positions with columns: Symbol, Shares

from __future__ import annotations

import pandas as pd


def _ensure_col(df: pd.DataFrame, col: str, default):
    """Ensure df has a column; if missing, create with default."""
    if col not in df.columns:
        df[col] = default
    return df


def _safe_numeric(series, default=0.0):
    """Convert to numeric safely; if scalar given, return scalar."""
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    # scalar
    try:
        return float(series)
    except Exception:
        return default


def _normalize_symbol(s: str) -> str:
    return str(s).strip().upper()


def apply_portfolio_context(screen_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach portfolio context columns to the screener dataframe:
    - OwnedShares
    - OwnedValue (OwnedShares * Price if Price exists else 0)
    - Weight (OwnedValue / total)
    """
    out = screen_df.copy()

    # Determine key in screener (Ticker preferred)
    key_col = "Ticker" if "Ticker" in out.columns else ("Symbol" if "Symbol" in out.columns else None)
    if key_col is None:
        # No key to merge; return safe defaults
        _ensure_col(out, "OwnedShares", 0.0)
        _ensure_col(out, "OwnedValue", 0.0)
        _ensure_col(out, "Weight", 0.0)
        return out

    # Prepare merge keys
    out[key_col] = out[key_col].astype(str).map(_normalize_symbol)

    pos = pos_df.copy()
    if "Symbol" not in pos.columns:
        # Can't use positions without symbol column
        _ensure_col(out, "OwnedShares", 0.0)
        _ensure_col(out, "OwnedValue", 0.0)
        _ensure_col(out, "Weight", 0.0)
        return out

    pos["Symbol"] = pos["Symbol"].astype(str).map(_normalize_symbol)
    if "Shares" not in pos.columns:
        pos["Shares"] = 0.0
    pos["Shares"] = _safe_numeric(pos["Shares"], 0.0)

    # Aggregate (in case multiple rows per symbol)
    pos_agg = pos.groupby("Symbol", as_index=False)["Shares"].sum()
    pos_agg.rename(columns={"Symbol": key_col, "Shares": "OwnedShares"}, inplace=True)

    # Merge into screener
    out = out.merge(pos_agg, on=key_col, how="left")
    out["OwnedShares"] = _safe_numeric(out.get("OwnedShares"), 0.0)

    # OwnedValue
    if "Price" in out.columns:
        out["Price"] = _safe_numeric(out["Price"], 0.0)
        out["OwnedValue"] = out["OwnedShares"] * out["Price"]
    else:
        out["OwnedValue"] = 0.0

    total = float(out["OwnedValue"].sum()) if "OwnedValue" in out.columns else 0.0
    if total > 0:
        out["Weight"] = out["OwnedValue"] / total
    else:
        out["Weight"] = 0.0

    return out


def decide_actions(screen_df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """
    Decide portfolio-aware Action: ADD / BUY / HOLD / TRIM / AVOID

    This uses robust defaults if rules are missing.

    Expected optional keys in rules (all optional):
      - max_position_weight: float (e.g., 0.08)
      - trim_weight: float (e.g., 0.06)  -> TRIM if weight >= this
      - add_weight: float (e.g., 0.02)   -> ADD if owned and weight < this and score/upside ok
      - min_upside_buy: float (e.g., 0.05)  -> BUY if not owned and upside >=
      - min_score_buy: float (e.g., 60)     -> BUY if not owned and score >=
      - avoid_if_no_data: bool (default True)
    """
    rules = rules or {}
    out = screen_df.copy()

    # Ensure columns exist
    _ensure_col(out, "OwnedShares", 0.0)
    _ensure_col(out, "Weight", 0.0)

    # Optional columns from screener
    _ensure_col(out, "Score", 0.0)
    _ensure_col(out, "UpsidePct", 0.0)

    # In case Score/UpsidePct came as scalar or object
    out["OwnedShares"] = _safe_numeric(out["OwnedShares"], 0.0)
    out["Weight"] = _safe_numeric(out["Weight"], 0.0)
    out["Score"] = _safe_numeric(out["Score"], 0.0)
    out["UpsidePct"] = _safe_numeric(out["UpsidePct"], 0.0)

    # PayoutRatio is optional; make it safe (never scalar fillna)
    if "PayoutRatio" in out.columns:
        out["PayoutRatio"] = _safe_numeric(out["PayoutRatio"], 0.0)
    else:
        out["PayoutRatio"] = 0.0

    # Defaults
    max_position_weight = float(rules.get("max_position_weight", 0.10))  # hard cap
    trim_weight = float(rules.get("trim_weight", 0.07))                  # TRIM threshold
    add_weight = float(rules.get("add_weight", 0.02))                    # ADD threshold
    min_upside_buy = float(rules.get("min_upside_buy", 0.05))
    min_score_buy = float(rules.get("min_score_buy", 60))
    avoid_if_no_data = bool(rules.get("avoid_if_no_data", True))

    # If your screener doesn’t compute Score realistically yet, BUY can be upside-driven.
    # We'll also avoid buying if both score and upside are missing/zero.
    def pick_action(row) -> str:
        owned = float(row.get("OwnedShares", 0.0)) > 0.0
        w = float(row.get("Weight", 0.0))
        score = float(row.get("Score", 0.0))
        upside = float(row.get("UpsidePct", 0.0))

        has_signal = (score != 0.0) or (upside != 0.0)

        # Overweight -> TRIM
        if owned and w >= trim_weight:
            return "TRIM"
        if owned and w >= max_position_weight:
            return "TRIM"

        # Underweight -> ADD (if any signal)
        if owned and w < add_weight and has_signal:
            return "ADD"

        # Not owned -> BUY if signal strong enough
        if not owned:
            if avoid_if_no_data and not has_signal:
                return "AVOID"
            if (score >= min_score_buy) or (upside >= min_upside_buy):
                return "BUY"
            return "HOLD"

        # Owned and not trim/add -> HOLD
        return "HOLD"

    out["Action"] = out.apply(pick_action, axis=1)
    return out
