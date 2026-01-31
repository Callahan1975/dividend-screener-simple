import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TICKERS = [
    # USA – Dividend / Quality core
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","HD","LOW","UNH",
    "ABBV","MRK","CVX","XOM","V","MA","TXN","AVGO","COST",

    # Canada
    "RY","TD","BMO","BNS","ENB","CNQ","FTS",

    # Europe (ADR / US tickers)
    "UL","BP","AZN","NVS"
]

OUTPUT_PATH = "data/screener_results.csv"

MAX_REASONABLE_YIELD = 15.0
MAX_REASONABLE_PAYOUT = 110.0

SECTOR_FAIR_PE = {
    "Technology": 22,
    "Healthcare": 20,
    "Consumer Defensive": 20,
    "Consumer Cyclical": 20,
    "Financial Services": 12,
    "Energy": 12,
    "Utilities": 16,
    "Industrials": 18,
    "Basic Materials": 14,
    "Real Estate": 16,
    "Communication Services": 18
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_float(v, pct=False):
    try:
        val = float(v)
        if pct:
            val *= 100
        return round(val, 2)
    except Exception:
        return None


def normalize_dividend_yield(raw):
    if raw is None:
        return None
    try:
        y = float(raw)
        if y <= 1:
            y *= 100
        if y <= 0 or y > MAX_REASONABLE_YIELD:
            return None
        return round(y, 2)
    except Exception:
        return None


def normalize_payout_ratio(raw):
    if raw is None:
        return None
    try:
        p = float(raw)
        if p <= 0 or p > MAX_REASONABLE_PAYOUT:
            return None
        return round(p, 2)
    except Exception:
        return None


def calc_dividend_cagr(dividends, years=5):
    if dividends is None or len(dividends) < years + 1:
        return None
    try:
        start = dividends.iloc[-years - 1]
        end = dividends.iloc[-1]
        if start <= 0 or end <= 0:
            return None
        return round(((end / sta
