from pathlib import Path
import pandas as pd

ROOT = Path(".")
ALIAS_FILE = ROOT / "data" / "ticker_alias.csv"
OUT_FILE = ROOT / "tickers.txt"

df = pd.read_csv(ALIAS_FILE, comment="#", skip_blank_lines=True)

tickers = (
    df["Ticker"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
    .unique()
)

OUT_FILE.write_text("\n".join(tickers) + "\n", encoding="utf-8")

print(f"✅ Generated tickers.txt with {len(tickers)} tickers")
