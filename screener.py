import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================
# Ticker-univers
# =========================

US_TICKERS = [
    "AAPL","MSFT","JNJ","PG","KO","PEP","MCD","WMT","COST","HD","LOW",
    "UNH","CVX","XOM","IBM","ADP","MMM","CL","KMB","ABT","ABBV",
    "T","VZ","CSCO","INTC","TXN","QCOM","AMGN","MDT","BDX","SYK",
    "UPS","FDX","CAT","DE","EMR","ETN","PH","HON","RTX","LMT",
    "NOC","GD","BA","GE","NEE","DUK","SO","D","AEP","ED",
    "SRE","EXC","XEL","PEG","WEC","DTE",
    "O","WPC","SPG","AVB","EQR","ESS","MAA","VTR","VICI",
    "TROW","BLK","BEN","AMP","MS","GS","JPM","BAC","PNC","USB",
    "SCHW","AFL","MET","PRU","ALL","TRV",
    "MO","PM","BTI",
    "WM","RSG","AWK","ATO","ECL","SHW","APD","LIN","NUE","VMC",
    "SBUX","YUM","CMI","ITW","ROP","FAST","GPC","ORCL","INTU"
]

CA_TICKERS = [
    "BMO.TO","RY.TO","TD.TO","BNS.TO","CM.TO","NA.TO",
    "ENB.TO","TRP.TO","PPL.TO",
    "FTS.TO","EMA.TO","CU.TO",
    "CNQ.TO","SU.TO","IMO.TO",
    "T.TO","BCE.TO","RCI-B.TO",
    "CP.TO","CNR.TO",
    "SLF.TO","MFC.TO","GWO.TO",
    "AQN.TO","KEY.TO","POW.TO","IFC.TO","BIP-UN.TO"
]

NORDIC_TICKERS = [
    "SHB-A.ST","SWED-A.ST","SEB-A.ST","NDA-SE.ST","TEL2-B.ST",
    "ATCO-A.ST","VOLV-B.ST","CIBUS.ST",
    "NOVO-B.CO","COLO-B.CO","ORSTED.CO","DSV.CO","TRYG.CO","PNDORA.CO",
    "NDA-FI.HE","KNEBV.HE","FORTUM.HE",
    "DNB.OL","ORK.OL"
]

TICKERS = US_TICKERS + CA_TICKERS + NORDIC_TICKERS


# =========================
# Hjælpefunktioner
# =========================

def pct(v):
    try:
        return round(float(v), 2)
    except:
        return 0.0


# =========================
# Screener
# =========================

rows = []

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        price = info.get("currentPrice")
        if not price:
            continue

        # ---------- Dividend Yield (TTM – robust) ----------
        dividend_yield = 0.0
        try:
            dividends = t.dividends
            if dividends is not None and len(dividends) > 0:
                ttm_div = dividends[
                    dividends.index >= (dividends.index.max() - pd.Timedelta(days=365))
                ].sum()
                dividend_yield = pct((ttm_div / price) * 100)
        except:
            dividend_yield = 0.0

        # ---------- Payout / ROE ----------
        payout = pct((info.get("payoutRatio") or 0) * 100)
        roe = pct((info.get("returnOnEquity") or 0) * 100)

        pe = info.get("trailingPE")
        eps = info.get("trailingEps")

        sector = info.get("sector") or ""

        # Fair PE (samme logik som før)
        fair_pe_map = {
            "Technology": 22,
            "Consumer Defensive": 20,
            "Healthcare": 20,
            "Industrials": 18,
            "Financial Services": 12,
            "Energy": 12,
            "Utilities": 16,
            "Real Estate": 16,
            "Basic Materials": 14,
            "Communication Services": 18
        }
        fair_pe = fair_pe_map.get(sector, 18)

        fair_value = eps * fair_pe if eps else None
        upside = pct(((fair_value / price) - 1) * 100) if fair_value else 0.0

        # ---------- Flags ----------
        flags = []
        if payout > 90:
            flags.append("Payout high")
        if payout > 110:
            flags.append("Payout extreme")

        # ---------- Signal ----------
        if upside >= 20 and payout <= 75:
            signal = "GOLD"
        elif upside >= 10 and payout <= 85:
            signal = "BUY"
        elif upside > 0:
            signal = "HOLD"
        else:
            signal = "WATCH"

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Country": info.get("country"),
            "Sector": sector,
            "Industry": info.get("industry"),
            "Price": round(price, 2),
            "DividendYield_%": dividend_yield,
            "PayoutRatio_%": payout,
            "ROE_%": roe,
            "Upside_%": upside,
            "Quality_ROE_10p": roe >= 10,
            "Signal": signal,
            "Flags": ", ".join(flags)
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

# =========================
# Output
# =========================

df = pd.DataFrame(rows)
df["GeneratedUTC"] = datetime.utcnow().isoformat()

df.to_csv("data/screener_results.csv", index=False)
df.to_csv("screener_results.csv", index=False)

print(f"Done. {len(df)} tickers processed.")
