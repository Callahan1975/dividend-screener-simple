from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    # Snowball exports can be ; or , and different encodings.
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            return df
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, sep=None, engine="python")


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = { _norm_col(c): c for c in df.columns }
    for cand in candidates:
        key = _norm_col(cand)
        if key in cols:
            return cols[key]
    # allow partial matches
    for cand in candidates:
        key = _norm_col(cand)
        for k, orig in cols.items():
            if key == k:
                return orig
            if key in k:
                return orig
    return None


def _load_aliases(alias_csv_path: Path) -> dict[str, str]:
    if alias_csv_path is None or not alias_csv_path.exists():
        return {}
    df = _read_csv_flexible(alias_csv_path)
    if df.empty:
        return {}
    # expected: alias,ticker OR from,to
    a = _find_col(df, ["alias", "from"])
    t = _find_col(df, ["ticker", "to"])
    if a is None or t is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        aa = str(r[a]).strip()
        tt = str(r[t]).strip()
        if aa and tt and aa.lower() != "nan" and tt.lower() != "nan":
            out[aa] = tt
    return out


def _apply_aliases(symbol: str, aliases: dict[str, str]) -> str:
    if not symbol:
        return symbol
    s = str(symbol).strip()
    return aliases.get(s, s)


def _detect_format(df: pd.DataFrame) -> str:
    cols = {_norm_col(c) for c in df.columns}

    # transactions-like
    if {"event", "quantity"}.issubset(cols) and ("symbol" in cols or "ticker" in cols):
        return "transactions"

    # holdings-like (many Snowball versions)
    if ("shares" in cols or "share" in cols or "antal" in cols) and ("holding" in cols or "symbol" in cols or "ticker" in cols):
        return "holdings"

    # another holdings export style
    if ("holding" in cols and ("shares" in cols or "share" in cols or "antal" in cols)):
        return "holdings"

    # some Snowball exports have these:
    if ("holding" in cols and ("current_value" in cols or "share_price" in cols or "kurs" in cols)):
        return "holdings"

    return "unknown"


def _positions_from_transactions(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    sym_col = _find_col(df, ["Symbol", "Ticker"])
    evt_col = _find_col(df, ["Event"])
    qty_col = _find_col(df, ["Quantity", "Qty", "Antal"])
    price_col = _find_col(df, ["Price", "Kurs", "Rate"])
    date_col = _find_col(df, ["Date", "Dato"])

    if sym_col is None or evt_col is None or qty_col is None:
        raise ValueError("Snowball transactions format missing required columns.")

    d = df.copy()
    d[sym_col] = d[sym_col].astype(str).str.strip()
    d[sym_col] = d[sym_col].apply(lambda x: _apply_aliases(x, aliases))

    d[qty_col] = pd.to_numeric(d[qty_col], errors="coerce").fillna(0.0)

    # BUY/SELL markers differ across exports; keep broad
    evt = d[evt_col].astype(str).str.lower()
    signed = d[qty_col].copy()
    signed[:] = 0.0

    buy_mask = evt.str.contains("buy") | evt.str.contains("køb") | evt.str.contains("purchase")
    sell_mask = evt.str.contains("sell") | evt.str.contains("salg") | evt.str.contains("sold")

    signed[buy_mask] = d.loc[buy_mask, qty_col]
    signed[sell_mask] = -d.loc[sell_mask, qty_col]

    # splits can appear; if split has ratio in price column etc. ignore (too messy) -> treat as 0
    d["SignedShares"] = signed

    if date_col is not None:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    pos = d.groupby(sym_col, dropna=False)["SignedShares"].sum().reset_index()
    pos.rename(columns={sym_col: "Ticker", "SignedShares": "Shares"}, inplace=True)
    pos = pos[pos["Shares"].abs() > 1e-9].copy()
    pos["Shares"] = pos["Shares"].astype(float)
    return pos[["Ticker", "Shares"]]


def _positions_from_holdings(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    sym_col = _find_col(df, ["Holding", "Symbol", "Ticker", "ISIN"])
    shares_col = _find_col(df, ["Shares", "Share", "Antal", "Quantity", "Units"])

    if sym_col is None:
        raise ValueError("Snowball holdings format missing ticker/holding column.")
    if shares_col is None:
        # try alternative: "Shares in portfolio" etc.
        shares_col = _find_col(df, ["Share_in_portfolio", "Shares_in_portfolio"])
    if shares_col is None:
        raise ValueError("Snowball holdings format missing shares/quantity column.")

    d = df.copy()
    d[sym_col] = d[sym_col].astype(str).str.strip()
    d[sym_col] = d[sym_col].apply(lambda x: _apply_aliases(x, aliases))

    d[shares_col] = pd.to_numeric(d[shares_col], errors="coerce").fillna(0.0)

    pos = d[[sym_col, shares_col]].copy()
    pos.rename(columns={sym_col: "Ticker", shares_col: "Shares"}, inplace=True)
    pos = pos[pos["Shares"].abs() > 1e-9].copy()
    pos["Shares"] = pos["Shares"].astype(float)
    return pos


def load_positions_from_snowball(
    snowball_csv_path: Path,
    alias_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Returns DataFrame: Ticker, Shares
    Safe: returns empty DF if file missing.
    """
    if snowball_csv_path is None or not Path(snowball_csv_path).exists():
        return pd.DataFrame(columns=["Ticker", "Shares"])

    snowball_csv_path = Path(snowball_csv_path)
    aliases = _load_aliases(Path(alias_csv_path)) if alias_csv_path else {}

    df = _read_csv_flexible(snowball_csv_path)
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Shares"])

    fmt = _detect_format(df)
    if fmt == "transactions":
        return _positions_from_transactions(df, aliases)
    if fmt == "holdings":
        return _positions_from_holdings(df, aliases)

    # fallback: try to guess
    try:
        return _positions_from_holdings(df, aliases)
    except Exception:
        pass
    try:
        return _positions_from_transactions(df, aliases)
    except Exception:
        pass

    raise ValueError(f"Unknown Snowball CSV format. Columns={list(df.columns)}")
