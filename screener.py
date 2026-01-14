import pandas as pd
from pathlib import Path

OUTPUT = Path("data/screener_results")
OUTPUT.mkdir(parents=True, exist_ok=True)

data = [
    ["MSFT", "Microsoft", "US", "Technology", 0.76, 0.23, 33],
    ["JNJ", "Johnson & Johnson", "US", "Healthcare", 2.9, 0.49, 20],
    ["PG", "Procter & Gamble", "US", "Consumer Defensive", 2.5, 0.60, 21],
    ["NOVO-B.CO", "Novo Nordisk", "DK", "Healthcare", 1.4, 0.50, 38],
]

df = pd.DataFrame(
    data,
    columns=[
        "Ticker",
        "Name",
        "Country",
        "Sector",
        "DividendYield_%",
        "PayoutRatio",
        "PE"
    ]
)

df.to_csv(OUTPUT / "screener_results.csv", index=False)
print("✅ screener_results.csv written")
