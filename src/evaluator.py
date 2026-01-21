import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "screener_results"

INPUT_FILE = DATA_DIR / "screener_results.csv"
OUTPUT_FILE = DATA_DIR / "screener_decision.csv"
DIV_CLASS_FILE = BASE_DIR / "data" / "dividend_classes.csv"

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Dividend classes
# -----------------------------
div_class = {}
if DIV_CLASS_FILE.exists():
    cdf = pd.read_csv(DIV_CLASS_FILE)
    div_class = dict(zip(cdf["Ticker"], cdf["DividendClass"]))

df["DividendClass"] = df["Ticker"].map(div_class).fillna("—")

# -----------------------------
# Score model
# -----------------------------
def score_row(r):
    s = 0

    y = r.get("DividendYield_%", 0)
    p = r.get("PayoutRatio_%", 0)
    pe = r.get("PE", None)
    dc = r.get("DividendClass", "")

    if 2 <= y <= 6: s += 15
    elif y > 6: s += 10

    if p < 60: s += 20
    elif p < 90: s += 10
    else: s -= 20

    if pe and pe < 20: s += 15
    elif pe and pe < 30: s += 5

    if dc == "King": s += 25
    elif dc == "Aristocrat": s += 20
    elif dc == "Contender": s += 10

    return max(0, min(100, s))

df["Score"] = df.apply(score_row, axis=1)

# -----------------------------
# Signal & Confidence
# -----------------------------
def signal(s):
    if s >= 85: return "GOLD"
    if s >= 70: return "BUY"
    if s >= 55: return "HOLD"
    return "WATCH"

df["Signal"] = df["Score"].apply(signal)

def confidence(r):
    if r["DividendClass"] in ["King", "Aristocrat"] and r["PayoutRatio_%"] < 80:
        return "High"
    if r["PayoutRatio_%"] >= 90:
        return "Low"
    return "Medium"

df["Conf"] = df.apply(confidence, axis=1)

# -----------------------------
# Why column
# -----------------------------
def why(r):
    parts = []
    if r["DividendClass"] != "—":
        parts.append(r["DividendClass"])
    if r["PayoutRatio_%"] >= 90:
        parts.append("Payout high")
    if r["DividendYield_%"] >= 6:
        parts.append("High yield")
    if r["PE"] and r["PE"] < 15:
        parts.append("Low PE")
    return ", ".join(parts) if parts else "—"

df["Why"] = df.apply(why, axis=1)

# -----------------------------
# Save
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Decision file written: {OUTPUT_FILE}")
