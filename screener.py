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
# Helpers
# =========================

def pct(v):
    try:
        return round(float(v), 2)
    except:
        return 0.0


def dividend_growth_metrics(divs: pd.Series):
    """
    Returns (cagr_5y, years_growing)
    """
    if divs is None or len(divs) < 5:
        return 0.0, 0

    # yearly totals
    yearly = divs.resample("Y").sum()
    yearly.index = yearly.index.year

    # drop current year (often incomplete)
    current_year = datetime.utcnow().year
    yearly = yearly[yearly.index < current_year]

    if len(yearly) < 5:
        return 0.0, 0

    # Years growing
    years = list(yearly.values)
    growing = 0
    for i in range(len(years) - 1, 0, -1):
        if years[i] > years[i - 1]:
            growing += 1
        else:
            break

    # 5Y CAGR
    try:
        start = yearly.iloc[-5]
        end = yearly.iloc[-1]
        cagr = ((end / start) ** (1 / 4) - 1) * 100 if start > 0 else 0.0
    except:
        cagr = 0.0

    return pct(cagr), growing


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

        # ---------- Dividend Yield (TTM approx) ----------
        dividend_yield = 0.0
        divs = t.dividends
        if divs is not None and len(divs) > 0:
            dividend_yield = pct((divs.tail(4).sum() / price) * 100)

        # ---------- Dividend Growth ----------
        div_cagr_5y, years_growing = dividend_growth_metrics(divs)

        payout = pct((info.get("payoutRatio") or 0) * 100)
        roe = pct((info.get("returnOnEquity") or 0) * 100)

        eps = info.get("trailingEps")
        sector = info.get("sector") or ""

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

        flags = []
        if payout > 90:
            flags.append("Payout high")
        if payout > 110:
            flags.append("Payout extreme")

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
            "DivCAGR_5Y_%": div_cagr_5y,
            "YearsGrowing": years_growing,
            "PayoutRatio_%": payout,
            "ROE_%": roe,
            "Upside_%": upside,
            "Quality_ROE_10p": roe >= 10,
            "Signal": signal,
            "Flags": ", ".join(flags)
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

df = pd.DataFrame(rows)
df["GeneratedUTC"] = datetime.utcnow().isoformat()

df.to_csv("data/screener_results.csv", index=False)
df.to_csv("screener_results.csv", index=False)

print(f"Done. {len(df)} tickers processed.")
