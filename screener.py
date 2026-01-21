import pandas as pd

# load data (din logik her)
df = pd.read_csv("raw_data.csv")

# HARD CONTRACT – kun disse kolonner
COLUMNS = [
    "Ticker",
    "Name",
    "Country",
    "Sector",
    "Price",
    "DividendYield",
    "PayoutRatio",
    "PE",
    "MarketCap"
]

df = df[COLUMNS]

# Ryd NaN
df = df.fillna("")

df.to_csv("data/screener_results.csv", index=False)
