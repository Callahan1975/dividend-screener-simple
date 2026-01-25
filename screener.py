import yfinance as yf
import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_FILE = os.path.join(BASE_DIR, "tickers.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "data", "screener_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screener_results.csv")

FIELDS = [
    "GeneratedUTC",
    "Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Price",
    "DividendYield","PayoutRatio","PE",
    "Confidence","Signal",
    "DividendClass",
    "Region",
    "Top10Region",
    "LTM_Dividend","LTM_Count"
]

DIVIDEND_CLASS = {
    "PG": "King","JNJ": "King","KO": "King","PEP": "King",
    "MMM": "King","EMR": "King","CL": "King","KMB": "King",
    "LOW": "King","HD": "King",

    "ABBV": "Aristocrat","ADP": "Aristocrat","AFL": "Aristocrat",
    "APD": "Aristocrat","CAT": "Aristocrat","CVX": "Aristocrat",
    "IBM": "Aristocrat","MCD": "Aristocrat","MSFT": "Aristocrat",
    "NEE": "Aristocrat",

    "AVGO": "Contender","COST": "Contender","TXN": "Contender",
    "WM": "Contender","UNH": "Contender","V": "Contender","MA": "Contender"
}

TOP10_REGIONS = {
    "DK": ["PNDORA.CO","NOVO-B.CO","TRYG.CO","CARL-B.CO","COLO-B.CO","DSV.CO","SYDB.CO","DANSKE.CO","ISS.CO","SPNO.CO"],
    "SE": ["SHB-A.ST","SEB-A.ST","SWED-A.ST","ATCO-A.ST","ATCO-B.ST","VOLV-B.ST","TEL2-B.ST","SBB-B.ST","CAST.ST","NDA-SE.ST"],
    "NL": ["UNA.AS","ASML.AS","NN.AS","INGA.AS","AD.AS","HEIA.AS","KPN.AS","ABN.AS","RAND.AS","DSM.AS"],
    "CA": ["ENB.TO","TD.TO","BMO.TO","BNS.TO","RY.TO","FTS.TO","TRP.TO","CNQ.TO","SU.TO","T.TO"]
}

def load_tickers():
    with open(TICKER_FILE, "r", encoding="utf-8") as f:
        return sorted({
            line.strip().split("#")[0].strip()
            for line in f
            if line.strip() and not line.startswith("#")
        })

def region_from_ticker(t):
    if t.endswith(".CO"): return "DK"
    if t.endswith(".ST"): return "SE"
    if t.endswith(".AS"): return "NL"
    if t.endswith(".TO"): return "CA"
    return "US"

def calc_signal(y, p):
    if y == "" or p == "": return "HOLD"
    if y >= 4 and p <= 70: return "BUY"
    if p > 90: return "AVOID"
    return "HOLD"

def calc_confidence(y, p, pe):
    s = 50
    if y != "" and y >= 3: s += 15
    if p != "" and p <= 70: s += 15
    if pe != "" and pe <= 20: s += 10
    return min(s, 100)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    gen = datetime.utcnow().isoformat()+"Z"

    for tkr in load_tickers():
        try:
            t = yf.Ticker(tkr)
            info = t.info
            price = info.get("regularMarketPrice")
            if not price: continue

            hist = t.history(period="400d", actions=True)
            divs = hist["Dividends"][hist["Dividends"] > 0] if "Dividends" in hist else []
            ltm = divs.sum() if len(divs) else ""
            yld = round((ltm/price)*100,2) if ltm != "" else ""

            eps = info.get("trailingEps")
            payout = round((ltm/eps)*100,2) if ltm != "" and eps and eps>0 else ""

            pe = round(info.get("trailingPE"),2) if info.get("trailingPE") else ""

            region = region_from_ticker(tkr)
            top10 = tkr in TOP10_REGIONS.get(region, [])

            rows.append({
                "GeneratedUTC": gen,
                "Ticker": tkr,
                "Name": info.get("longName",""),
                "Country": info.get("country",""),
                "Currency": info.get("currency",""),
                "Exchange": info.get("exchange",""),
                "Sector": info.get("sector",""),
                "Industry": info.get("industry",""),
                "Price": round(price,2),
                "DividendYield": yld,
                "PayoutRatio": payout,
                "PE": pe,
                "Confidence": calc_confidence(yld,payout,pe),
                "Signal": calc_signal(yld,payout),
                "DividendClass": DIVIDEND_CLASS.get(tkr,""),
                "Region": region,
                "Top10Region": "Yes" if top10 else "No",
                "LTM_Dividend": round(ltm,2) if ltm!="" else "",
                "LTM_Count": len(divs)
            })

        except Exception as e:
            print("Skip", tkr, e)

    with open(OUTPUT_FILE,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
