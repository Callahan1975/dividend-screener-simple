import pandas as pd
from datetime import datetime

data = [
    ["AAPL", "Apple Inc.", "United States", "USD", "NASDAQ",
     "Technology", "Consumer Electronics", 247.65, 0.52, 13.7, 33.2, 48, "WATCH"],

    ["ABBV", "AbbVie Inc.", "United States", "USD", "NYSE",
     "Healthcare", "Drug Manufacturers", 216.15, 3.13, 49.9, 16.5, 72, "HOLD"],

    ["AEP", "American Electric Power", "United States", "USD", "NYSE",
     "Utilities", "Electric Utilities", 116.62, 3.26, 54.5, 17.1, 82, "BUY"],

    ["CARL-B.CO", "Carlsberg A/S", "Denmark", "DKK", "CPH",
     "Consumer Defensive", "Beverages", 848.00, 3.23, 53.6, 16.9, 78, "BUY"],
]

columns = [
    "Ticker", "Name", "Country", "Currency", "Exchange",
    "Sector", "Industry", "Price",
    "DividendYield", "PayoutRatio", "PE", "Confidence", "Signal"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("docs/screener_results.csv", index=False)
print("✅ screener_results.csv written")
