import os
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

# Optional portfolio modules (if present in repo)
PORTFOLIO_AVAILABLE = True
try:
    import yaml
    from src.portfolio_ingest import build_positions
    from src.portfolio_actions import apply_portfolio_context, decide_actions
except Exception:
    PORTFOLIO_AVAILABLE = False


# -------------------------
# CONFIG
# -------------------------
TICKERS_FILE = "tickers.txt"

SNOWBALL_PATH = "data/portfolio/Snowball.csv"
RULES_PATH = "config/portfolio_rules.yml"
ALIAS_PATH = "data/portfolio/ticker_alias.csv"

OUT_DIR = Path("data/screener_results")
OUT_FULL = OUT_DIR / "screener_results.csv"
OUT_SIMPLE = OUT_DIR / "simple_screener_results.csv"

# Valuation model knobs (simple, stable)
DISCOUNT_RATE = 0.10   # DDM discount rate
MAX_G = 0.08           # cap growth used in DDM
MIN_G = 0.00           # floor growth
YIELD_MODEL_FLOOR = 0.012  # 1.2%
YIELD_MODEL_CAP = 0.08     # 8.0%

# yfinance throttling
SLEEP_BETWEEN = 0.15
MAX_RETRIES = 2


# -------------------------
# HELPERS
# -------------------------
def read_tickers(path: str) -> List[str]:
    tickers: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s.split()[0])
    return sorted(set(tickers))


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return (a - b) / b


def _try_get_info(ticker: yf.Ticker) -> Dict[str, Any]:
    # yfinance sometimes fails on .info; keep it safe
    for _ in range(MAX_RETRIES + 1):
        try:
            info = ticker.info or {}
            if isinstance(info, dict):
                return info
        except Exception:
            time.sleep(0.25)
    return {}


def _try_get_fast_price(info: Dict[str, Any]) -> Optional[float]:
    return safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or info.get("open")
    )


def _annual_dividend_from_info(info: Dict[str, Any]) -> Optional[float]:
    # Most stable field when available
    return safe_float(info.get("dividendRate"))


def _dividend_yield_from_info(info: Dict[str, Any]) -> Optional[float]:
    # Ratio: 0.03 = 3%
    y = safe_float(info.get("dividendYield"))
    if y is None:
        return None
    # Sometimes comes as 3 instead of 0.03
    if abs(y) > 1.0:
        y = y / 100.0
    return y


def _get_dividends_series(ticker: yf.Ticker) -> Optional[pd.Series]:
    try:
        d = ticker.dividends
        if d is None or len(d) == 0:
            return None
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        d = d.dropna()
        if len(d) == 0:
            return None
        return d
    except Exception:
        return None


def dividend_cagr_5y(div_series: Optional[pd.Series]) -> Optional[float]:
    """
    Estimate 5Y dividend CAGR using annual dividend sums.
    Needs at least 6 calendar years of data points (start + end).
    """
    if div_series is None or len(div_series) == 0:
        return None

    try:
        yearly = div_series.resample("Y").sum()
        yearly.index = yearly.index.year
        yearly = yearly[yearly > 0]

        if len(yearly) < 6:
            return None

        last_year = int(yearly.index.max())
        start_year = last_year - 5

        if start_year not in yearly.index or last_year not in yearly.index:
            return None

        start = float(yearly.loc[start_year])
        end = float(yearly.loc[last_year])

        if start <= 0 or end <= 0:
            return None

        cagr = (end / start) ** (1 / 5) - 1
        # sanity clamp
        if cagr < -0.5 or cagr > 0.5:
            return None
        return cagr
    except Exception:
        return None


def fair_value_yield(price: Optional[float], annual_div: Optional[float], yield_for_model: Optional[float]) -> Optional[float]:
    if price is None or annual_div is None or yield_for_model is None:
        return None
    if annual_div <= 0 or yield_for_model <= 0:
        return None
    return annual_div / yield_for_model


def fair_value_ddm(annual_div: Optional[float], g: Optional[float]) -> Optional[float]:
    """
    Simple Gordon Growth Model:
      FV = D1 / (r - g) where D1 = D0*(1+g)
    """
    if annual_div is None or annual_div <= 0:
        return None
    if g is None:
        g = 0.0
    g = clamp(g, MIN_G, MAX_G)

    r = DISCOUNT_RATE
    # avoid division by zero / negative
    if r <= g + 0.005:
        g = r - 0.005
    if r <= 0:
        return None

    d1 = annual_div * (1 + g)
    return d1 / (r - g)


def upside_pct(price: Optional[float], fv: Optional[float]) -> Optional[float]:
    if price is None or fv is None:
        return None
    if price <= 0:
        return None
    return (fv - price) / price


def score_row(upside: Optional[float], g5: Optional[float], pe: Optional[float]) -> float:
    """
    Stable scoring: combine upside + growth + valuation sanity.
    Returns ~0..5 range typically.
    """
    u = upside if upside is not None else 0.0
    g = g5 if g5 is not None else 0.0
    # clamp extremes
    u = clamp(u, -0.5, 1.0)     # -50% .. +100%
    g = clamp(g, -0.1, 0.2)     # -10% .. +20%

    pe_penalty = 0.0
    if pe is not None and pe > 0:
        # mild penalty for very high PE
        if pe >= 35:
            pe_penalty = 0.6
        elif pe >= 25:
            pe_penalty = 0.3

    # scale: upside up to 2.5pts, growth up to 1.5pts, then penalty
    score = (u * 2.5) + (g * 7.5)  # growth: 0.2*7.5=1.5
    score = score - pe_penalty
    # shift to a positive-ish baseline
    score = score + 1.5
    return float(round(score, 3))


def reco_from_score(score: float) -> str:
    if score >= 3.3:
        return "GOLD"
    if score >= 2.6:
        return "BUY"
    if score >= 1.9:
        return "HOLD"
    if score >= 1.3:
        return "WATCH"
    return "AVOID"


def action_from_upside(up: Optional[float], score: float) -> str:
    """
    Base action (not portfolio-aware). Portfolio rules can overwrite later.
    """
    u = up if up is not None else 0.0
    if u >= 0.25 and score >= 2.6:
        return "BUY"
    if u >= 0.10 and score >= 2.1:
        return "ADD"
    if u <= -0.15 and score >= 2.0:
        return "TRIM"
    if score < 1.3:
        return "AVOID"
    return "HOLD"


def load_alias_map() -> Dict[str, str]:
    if not os.path.exists(ALIAS_PATH):
        return {}
    try:
        a = pd.read_csv(ALIAS_PATH)
        a.columns = [c.strip() for c in a.columns]
        if len(a.columns) < 2:
            return {}
        src, dst = a.columns[0], a.columns[1]
        m = dict(zip(a[src].astype(str).str.strip(), a[dst].astype(str).str.strip()))
        return m
    except Exception:
        return {}


def apply_aliases_to_positions(pos_df: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    if pos_df is None or len(pos_df) == 0 or not alias_map:
        return pos_df
    out = pos_df.copy()
    if "Symbol" in out.columns:
        out["Symbol"] = out["Symbol"].astype(str).str.strip().map(lambda s: alias_map.get(s, s))
    return out


# -------------------------
# MAIN
# -------------------------
def build_screener_df(tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for t in tickers:
        time.sleep(SLEEP_BETWEEN)
        tk = yf.Ticker(t)
        info = _try_get_info(tk)

        name = info.get("shortName") or info.get("longName") or ""
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        country = info.get("country") or info.get("region") or ""

        price = _try_get_fast_price(info)
        div_yield = _dividend_yield_from_info(info)
        annual_div = _annual_dividend_from_info(info)
        payout = safe_float(info.get("payoutRatio"))
        pe = safe_float(info.get("trailingPE"))

        divs = _get_dividends_series(tk)
        g5 = dividend_cagr_5y(divs)

        # Yield model input
        yield_for_model = div_yield if div_yield is not None else None
        if yield_for_model is not None:
            yield_for_model = clamp(yield_for_model, YIELD_MODEL_FLOOR, YIELD_MODEL_CAP)

        fv_y = fair_value_yield(price, annual_div, yield_for_model)
        fv_ddm = fair_value_ddm(annual_div, g5)

        up_y = upside_pct(price, fv_y)
        up_ddm = upside_pct(price, fv_ddm)

        # Combine upside as max of available
        up_best = None
        candidates = [x for x in [up_y, up_ddm] if x is not None]
        if candidates:
            up_best = max(candidates)

        score = score_row(up_best, g5, pe)
        reco = reco_from_score(score)
        action = action_from_upside(up_best, score)

        rows.append({
            "Ticker": t,
            "Name": name,
            "Country": country,
            "Sector": sector,
            "Industry": industry,

            "Price": price,
            "DividendYield": div_yield,     # ratio
            "AnnualDividend": annual_div,
            "PayoutRatio": payout,
            "PE": pe,

            "DividendGrowth5Y": g5,         # ratio
            "YieldForModel": yield_for_model,

            "FairValue_Yield": fv_y,
            "FairValue_DDM": fv_ddm,

            "Upside_Yield_%": up_y,
            "Upside_DDM_%": up_ddm,
            "UpsidePct": up_best,

            "Score": score,
            "Reco": reco,
            "Action": action,
        })

    df = pd.DataFrame(rows)

    # Display-friendly percent columns
    def to_pct_str(v: Any, digits=1) -> str:
        x = safe_float(v)
        if x is None:
            return ""
        return f"{x*100:.{digits}f}%"

    df["YieldPct"] = df["DividendYield"].apply(lambda x: "" if safe_float(x) is None else f"{float(x)*100:.2f}%")
    df["DivGrowth5YPct"] = df["DividendGrowth5Y"].apply(lambda x: to_pct_str(x, 1))
    df["UpsideBestPct"] = df["UpsidePct"].apply(lambda x: to_pct_str(x, 1))

    return df


def apply_portfolio_if_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Snowball + rules exist and modules are available, enrich with OwnedShares/OwnedValue/Weight
    and compute PortfolioAction via rules.
    """
    out = df.copy()

    # Keep original action
    out["ScreenerAction"] = out["Action"]

    if not PORTFOLIO_AVAILABLE:
        out["OwnedShares"] = 0
        out["OwnedValue"] = 0
        out["Weight"] = 0
        out["PortfolioAction"] = "HOLD"
        return out

    if not (os.path.exists(SNOWBALL_PATH) and os.path.exists(RULES_PATH)):
        out["OwnedShares"] = 0
        out["OwnedValue"] = 0
        out["Weight"] = 0
        out["PortfolioAction"] = "HOLD"
        return out

    # Load positions
    pos_df = build_positions(SNOWBALL_PATH)
    alias_map = load_alias_map()
    pos_df = apply_aliases_to_positions(pos_df, alias_map)

    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}

    # Ensure we have UpsidePct as ratio for rules
    if "UpsidePct" not in out.columns:
        out["UpsidePct"] = 0.0
    out["UpsidePct"] = out["UpsidePct"].apply(lambda x: safe_float(x, 0.0) or 0.0)

    # Apply portfolio context + decision rules
    out = apply_portfolio_context(out, pos_df)
    out = decide_actions(out, rules)

    out["PortfolioAction"] = out["Action"]
    out["Action"] = out["PortfolioAction"]
    return out


def write_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full
    df.to_csv(OUT_FULL, index=False, encoding="utf-8")

    # Simple (slim view for dashboard)
    simple_cols = [
        "Ticker", "Name",
        "PortfolioAction", "Action", "ScreenerAction", "Reco", "Score",
        "YieldPct", "DivGrowth5YPct", "UpsideBestPct",
        "Country", "Sector", "Industry",
        "Weight", "OwnedValue", "OwnedShares",
        "Price"
    ]
    keep = [c for c in simple_cols if c in df.columns]
    df_simple = df[keep].copy()
    df_simple.to_csv(OUT_SIMPLE, index=False, encoding="utf-8")


def main():
    tickers = read_tickers(TICKERS_FILE)
    df = build_screener_df(tickers)
    df2 = apply_portfolio_if_available(df)

    # Stable sorting: PortfolioAction then Score desc
    if "Score" in df2.columns:
        df2 = df2.sort_values(by=["Action", "Score"], ascending=[True, False], kind="mergesort")

    write_outputs(df2)
    print(f"Wrote: {OUT_FULL}")
    print(f"Wrote: {OUT_SIMPLE}")


if __name__ == "__main__":
    main()
