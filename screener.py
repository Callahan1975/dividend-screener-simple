import yfinance as yf
import csv, os, math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_FILE = os.path.join(BASE_DIR, "tickers.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "data", "screener_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screener_results.csv")

FIELDS = [
    "GeneratedUTC","Ticker","Name","Country","Currency","Exchange",
    "Sector","Industry","Region","DividendClass","Top10Region",
    "Price","DividendYield","PayoutRatio","SectorPayoutMax",
    "DivCAGR_3Y","DivCAGR_5Y",
    "PE","Confidence","Signal",
    "LTM_Dividend","LTM_Count"
]

SECTOR_PAYOUT_MAX = {
    "Utilities":110,
    "Financial Services":120,
    "Energy":130,
    "Real Estate":150,
    "Consumer Defensive":90,
    "Industrials":80,
    "Technology":70,
    "Healthcare":80,
    "Basic Materials":80,
    "Consumer Cyclical":80,
    "Default":75
}

DIVIDEND_CLASS = {
    "PG":"King","JNJ":"King","KO":"King","PEP":"King","EMR":"King","CL":"King",
    "KMB":"King","LOW":"King","HD":"King",
    "ABBV":"Aristocrat","ADP":"Aristocrat","AFL":"Aristocrat","APD":"Aristocrat",
    "CAT":"Aristocrat","CVX":"Aristocrat","IBM":"Aristocrat","MCD":"Aristocrat",
    "MSFT":"Aristocrat","NEE":"Aristocrat",
    "AVGO":"Contender","COST":"Contender","TXN":"Contender","WM":"Contender",
    "UNH":"Contender","V":"Contender","MA":"Contender"
}

def load_tickers():
    with open(TICKER_FILE,"r",encoding="utf-8") as f:
        return sorted({
            l.split("#")[0].strip()
            for l in f if l.strip() and not l.startswith("#")
        })

def region_from_ticker(t):
    if t.endswith(".CO"): return "DK"
    if t.endswith(".ST"): return "SE"
    if t.endswith(".AS"): return "NL"
    if t.endswith(".TO"): return "CA"
    return "US"

def calc_ltm_and_cagr(t):
    hist = t.history(period="8y", actions=True)
    if hist.empty or "Dividends" not in hist:
        return ("",0,"","")
    divs = hist["Dividends"]
    yearly = divs[divs>0].groupby(divs.index.year).sum()

    ltm = yearly.iloc[-1] if len(yearly)>=1 else ""
    count = (divs>0).sum()

    def cagr(years):
        if len(yearly) < years+1: return ""
        start = yearly.iloc[-(years+1)]
        end = yearly.iloc[-1]
        if start<=0 or end<=0: return ""
        return round((end/start)**(1/years)-1,4)

    return (
        round(float(ltm),2) if ltm!="" else "",
        int(count),
        cagr(3),
        cagr(5)
    )

def calc_confidence(y,p,pe,c3,c5):
    score = 50
    if y>=3: score+=15
    if c5!="" and c5>=0.05: score+=15
    elif c3!="" and c3>=0.05: score+=10
    if p!="" and p<=70: score+=10
    if pe!="" and pe<=20: score+=5
    return min(score,100)

def calc_signal(y,p,pmax,c3,c5,conf):
    if p!="" and p>pmax: return "AVOID"
    if (c5!="" and c5<0) or (c3!="" and c3<0): return "AVOID"
    if y>=3 and conf>=70 and ((c5!="" and c5>=0.03) or (c3!="" and c3>=0.03)):
        return "BUY"
    return "HOLD"

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    rows=[]
    gen=datetime.utcnow().isoformat()+"Z"

    for tkr in load_tickers():
        try:
            t=yf.Ticker(tkr)
            info=t.info
            price=info.get("regularMarketPrice")
            if not price: continue

            ltm, cnt, c3, c5 = calc_ltm_and_cagr(t)
            y=round((ltm/price)*100,2) if ltm!="" else ""
            eps=info.get("trailingEps")
            p=round((ltm/eps)*100,2) if ltm!="" and eps and eps>0 else ""

            sector=info.get("sector","")
            pmax=SECTOR_PAYOUT_MAX.get(sector,SECTOR_PAYOUT_MAX["Default"])
            pe=round(info.get("trailingPE"),2) if info.get("trailingPE") else ""

            conf=calc_confidence(y,p,pe,c3,c5)
            sig=calc_signal(y,p,pmax,c3,c5,conf)

            rows.append({
                "GeneratedUTC":gen,
                "Ticker":tkr,
                "Name":info.get("longName",""),
                "Country":info.get("country",""),
                "Currency":info.get("currency",""),
                "Exchange":info.get("exchange",""),
                "Sector":sector,
                "Industry":info.get("industry",""),
                "Region":region_from_ticker(tkr),
                "DividendClass":DIVIDEND_CLASS.get(tkr,""),
                "Top10Region":"Yes" if region_from_ticker(tkr)!="US" else "No",
                "Price":round(price,2),
                "DividendYield":y,
                "PayoutRatio":p,
                "SectorPayoutMax":pmax,
                "DivCAGR_3Y":c3,
                "DivCAGR_5Y":c5,
                "PE":pe,
                "Confidence":conf,
                "Signal":sig,
                "LTM_Dividend":ltm,
                "LTM_Count":cnt
            })
        except Exception as e:
            print("Skip",tkr,e)

    with open(OUTPUT_FILE,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

if __name__=="__main__":
    main()
