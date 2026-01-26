import argparse
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

data = [
    ["AAPL", "Apple Inc.", "United States", "Technology", "Consumer Electronics", 261.05, 0.40, 13.67, 34.95],
    ["JNJ", "Johnson & Johnson", "United States", "Healthcare", "Drug Manufacturers - General", 213.65, 2.43, 49.08, 20.66],
    ["NOVO-B.CO", "Novo Nordisk A/S", "Denmark", "Healthcare", "Drug Manufacturers - General", 381.35, 3.04, 49.94, 16.34],
    ["DANSKE.CO", "Danske Bank A/S", "Denmark", "Financial Services", "Banks - Regional", 321.80, 5.85, 34.25, 11.78],
]

df = pd.DataFrame(
    data,
    columns=[
        "Ticker",
        "Name",
        "Country",
        "Sector",
        "Industry",
        "Price",
        "Dividend Yield (%)",
        "Payout Ratio (%)",
        "PE",
    ],
)

df.insert(0, "GeneratedUTC", now)

df.to_csv(args.output, index=False)
print(f"Saved screener results to {args.output}")
