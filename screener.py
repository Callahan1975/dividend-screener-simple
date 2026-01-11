#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dividend Screener v2 (DK + US)
Authoritative, stable, decision-focused screener
"""

from __future__ import annotations
import json, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

OUT_CSV = DATA_DIR / "screener_results.csv"
OUT_HTML = DOCS_DIR / "index.html"

BATCH = 40
SLEEP = 1.0

NORMAL_YIELD = 0.03
DISCOUNT = 0.08
SPECIAL_YIELD = 0.12
YIELD_CAP = 0.12

# ---------------- HELPERS ----------------
def sf(x):
    try:
        if x is None or pd.isna(x): return None
        return float(x)
    except: return None

def norm_frac(x):
    if x is None: return None
    return x / 100 if x > 1 else x

# ---------------- INPUT ----------------
def read_tickers():
    for p in ["tickers.txt", "data/tickers.txt"]:
        if Path(p).exists():
            t = [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
            return list(dict.fromkeys(t)), p
    raise SystemExit("Missing tickers.txt")

def read_index_map():
    p = Path("index_map.csv")
    if not p.exists(): return {}
    df = pd.read_csv(p)
    return dict(zip(df["Ticker"], df["Indexes"]))

# ---------------- MODELS ----------------
def score(y, g, pe):
    sy = min((y or 0)/0.06*100,100)
    sg = min((g or 0)/0.15*100,100)
    sv = 40 if not pe or pe<=0 else min((30-pe)/20*100,100)
    return round(0.35*sy+0.35*sg+0.30*sv,2)

def fair_yield(p,y): return round((p*y)/NORMAL_YIELD,2) if p and y else None
def fair_ddm(p,y,g):
    if not p or not y: return None
    g=min(g or 0.02,0.06)
    if DISCOUNT<=g: return None
    return round((p*y*(1+g))/(DISCOUNT-g),2)

def upside(p,fv): return round((fv/p-1)*100,1) if p and fv else None

def action(u,s,special):
    if special: return "VENT"
    if u is None or s is None: return "VENT"
    if u>=15 and s>=75: return "KØB NU"
    if u<5: return "FOR DYR"
    return "VENT"

# ---------------- FETCH ----------------
def fetch(tickers):
    T=yf.Tickers(" ".join(tickers))
    rows=[]
    for t in tickers:
        i=T.tickers[t].info
        p=sf(i.get("currentPrice"))
        y=sf(i.get("dividendYield"))
        y=norm_frac(y)
        g=None
        try:
            d=T.tickers[t].dividends
            if len(d)>=6:
                yr=d.resample("Y").sum()
                g=(yr.iloc[-1]/yr.iloc[-6])**(1/5)-1
        except: pass
        rows.append({
            "Ticker":t,"Name":i.get("shortName",""),
            "Price":p,"Currency":i.get("currency",""),
            "PE":sf(i.get("trailingPE")),
            "DividendYield":y,
            "DividendGrowth5Y":g,
            "Sector":i.get("sector",""),
            "Country":i.get("country","")
        })
    return pd.DataFrame(rows)

# ---------------- HTML ----------------
def build_html(df, gen, src):
    data=json.dumps(df.replace({np.nan:""}).to_dict("records"))
    return f"""<!doctype html><html><head>
<meta charset=utf-8><title>Dividend Screener</title>
<style>
body{{background:#0b0f14;color:#e6edf3;font-family:sans-serif}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #223042}}
.right{{text-align:right}}
.pill{{padding:3px 10px;border-radius:999px}}
.green{{background:#1aa95933}}
.amber{{background:#ffc84033}}
.gray{{background:#94a3b420}}
</style></head><body>
<h2>Dividend Screener (DK + US)</h2>
<p>Generated {gen} • Source {src}</p>
<table><thead><tr>
<th>Ticker</th><th>Name</th><th class=right>Price</th><th>CCY</th>
<th class=right>Yield</th><th class=right>FV (Yield)</th>
<th class=right>Upside %</th><th>Action</th>
</tr></thead><tbody id=b></tbody></table>
<script>
const D={data};
const b=document.getElementById("b");
D.forEach(r=>{
 const tr=document.createElement("tr");
 const a=r.Action||"";
 const cls=a==="KØB NU"?"green":a==="VENT"?"amber":"gray";
 tr.innerHTML=`
<td>${{r.Ticker}}</td>
<td>${{r.Name}}</td>
<td class=right>${{r.Price?.toFixed(2)||""}}</td>
<td>${{r.Currency}}</td>
<td class=right>${{(r.DividendYield*100).toFixed(1)}}%</td>
<td class=right>${{r.FairValue_Yield||""}}</td>
<td class=right>${{r.Upside_Yield_%||""}}%</td>
<td><span class="pill ${{cls}}">${{a}}</span></td>`;
 b.appendChild(tr);
});
</script></body></html>"""

# ---------------- MAIN ----------------
def main():
    tickers,src=read_tickers()
    idx=read_index_map()
    frames=[]
    for i in range(0,len(tickers),BATCH):
        frames.append(fetch(tickers[i:i+BATCH]))
        time.sleep(SLEEP)
    df=pd.concat(frames)
    df["Indexes"]=df["Ticker"].map(idx).fillna("")
    df["Special"]=df["DividendYield"].apply(lambda y:y and y>SPECIAL_YIELD)
    df["YieldModel"]=df["DividendYield"].apply(lambda y:min(y,YIELD_CAP) if y else None)
    df["Score"]=df.apply(lambda r:score(r.YieldModel,r.DividendGrowth5Y,r.PE),1)
    df["FairValue_Yield"]=df.apply(lambda r:fair_yield(r.Price,r.YieldModel),1)
    df["FairValue_DDM"]=df.apply(lambda r:fair_ddm(r.Price,r.YieldModel,r.DividendGrowth5Y),1)
    df["Upside_Yield_%"]=df.apply(lambda r:upside(r.Price,r.FairValue_Yield),1)
    df["Action"]=df.apply(lambda r:action(r.Upside_Yield_%,r.Score,r.Special),1)
    df.to_csv(OUT_CSV,index=False)
    OUT_HTML.write_text(build_html(df,dt.datetime.now(),src))
    print("DONE")

if __name__=="__main__": main()
