import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "screener_results"
INPUT_FILE = DATA_DIR / "screener_results.csv"
CLASS_FILE = BASE_DIR / "data" / "dividend_classes.csv"
OUTPUT_FILE = DATA_DIR / "screener_decision.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

classes = {}
if CLASS_FILE.exists():
    cls_df = pd.read_csv(CLASS_FILE)
    classes = dict(zip(cls_df["Ticker"], cls_df["DividendClass"]))

# =========================
# HELPERS
# =========================
def safe(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def score_row(r):
    score = 0
    reasons = []

    y = safe(r["DividendYield_%"])
    p = safe(r["PayoutRatio_%"])
    pe = safe(r["PE"])

    # Yield
    if y:
        if 2 <= y <= 6:
            score += 15
            reasons.append("Good yield")
        elif y > 6:
            score += 10
            reasons.append("High yield")

    # Payout
    if p:
        if p < 60:
            score += 20
            reasons.append("Low payout")
        elif p <= 90:
            score += 10
        else:
            score -= 20
            reasons.append("Payout risk")

    # PE
    if pe:
        if pe < 20:
            score += 15
        elif pe <= 30:
            score += 5

    # Dividend class
    cls = r.get("DividendClass")
    if cls == "King":
        score += 25
        reasons.append("Dividend King")
    elif cls == "Aristocrat":
        score += 20
        reasons.append("Aristocrat")
    elif cls == "Contender":
        score += 10

    return max(0, min(100, score)), ", ".join(reasons)

def signal(score):
    if score >= 85:
        return "GOLD"
    if score >= 70:
        return "BUY"
    if score >= 55:
        return "HOLD"
    return "WATCH"

def confidence(row):
    if "Payout risk" in row["Why"]:
        return "Low"
    if row["DividendClass"] in ("King", "Aristocrat"):
        return "High"
    return "Medium"

# =========================
# ENRICH DATA
# =========================
df["DividendClass"] = df["Ticker"].map(classes).fillna("")
scores = df.apply(score_row, axis=1, result_type="expand")
df["Score"] = scores[0]
df["Why"] = scores[1]
df["Signal"] = df["Score"].apply(signal)
df["Confidence"] = df.apply(confidence, axis=1)

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Decision file written: {OUTPUT_FILE}")

