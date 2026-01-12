import pandas as pd

TRANSACTION_EVENTS_BUY = {"BUY", "Buy", "PURCHASE", "Purchase"}
TRANSACTION_EVENTS_SELL = {"SELL", "Sell", "SALE", "Sale"}
TRANSACTION_EVENTS_SPLIT = {"SPLIT", "Split"}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def load_snowball_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_cols(df)

def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    # Snowball transactions template
    if {"Event", "Date", "Symbol", "Price", "Quantity", "Currency"}.issubset(cols):
        return "transactions"
    # holdings-like export
    if ("Symbol" in cols or "Ticker" in cols) and (("Shares" in cols) or ("Quantity" in cols)):
        return "holdings"
    return "unknown"

def positions_from_holdings(df: pd.DataFrame) -> pd.DataFrame:
    symbol_col = "Symbol" if "Symbol" in df.columns else "Ticker"
    qty_col = "Shares" if "Shares" in df.columns else "Quantity"

    pos = df[[symbol_col, qty_col]].rename(columns={symbol_col: "Symbol", qty_col: "Shares"}).copy()
    pos["Symbol"] = pos["Symbol"].astype(str).str.strip()
    pos["Shares"] = pd.to_numeric(pos["Shares"], errors="coerce").fillna(0.0)

    pos = pos.groupby("Symbol", as_index=False)["Shares"].sum()
    pos = pos[pos["Shares"].abs() > 1e-9]
    return pos

def positions_from_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Event"] = df["Event"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    df["SignedShares"] = 0.0
    df.loc[df["Event"].isin(TRANSACTION_EVENTS_BUY), "SignedShares"] = df["Quantity"]
    df.loc[df["Event"].isin(TRANSACTION_EVENTS_SELL), "SignedShares"] = -df["Quantity"]

    tx = df[["Date", "Event", "Symbol", "SignedShares", "Price"]].copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.sort_values(["Symbol", "Date"])

    holdings = {}
    for row in tx.itertuples(index=False):
        sym = row.Symbol
        holdings.setdefault(sym, 0.0)

        # Split handling (simple): multiplier is in Price
        if row.Event in TRANSACTION_EVENTS_SPLIT and row.Price > 0:
            holdings[sym] = holdings[sym] * float(row.Price)
        else:
            holdings[sym] = holdings[sym] + float(row.SignedShares)

    pos = pd.DataFrame({"Symbol": list(holdings.keys()), "Shares": list(holdings.values())})
    pos = pos[pos["Shares"].abs() > 1e-9]
    return pos

def build_positions(path: str) -> pd.DataFrame:
    df = load_snowball_csv(path)
    fmt = detect_format(df)
    if fmt == "transactions":
        return positions_from_transactions(df)
    if fmt == "holdings":
        return positions_from_holdings(df)
    raise ValueError(f"Unknown Snowball CSV format. Columns={list(df.columns)}")
