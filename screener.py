import pandas as pd
from pathlib import Path

# ---- CONTRACT PATH ----
CSV_PATH = Path("data/screener_results.csv")

# ---- VALIDATE CONTRACT ----
if not CSV_PATH.exists():
    raise FileNotFoundError(
        "❌ Contract CSV missing: data/screener_results.csv"
    )

# ---- LOAD CONTRACT ----
df = pd.read_csv(CSV_PATH)

# ---- BASIC SANITY CHECK ----
if df.empty:
    raise ValueError("❌ Contract CSV is empty")

# ---- RE-SAVE (NORMALIZE) ----
df.to_csv(CSV_PATH, index=False)

print("✅ Dividend Screener contract CSV validated and preserved")
