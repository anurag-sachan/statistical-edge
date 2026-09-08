from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import csv

START = date(2025, 12, 4)
END   = date(2026, 9, 7)

BASE = "https://data.binance.vision/data/spot/daily/klines/BTCUSD/2h"
OUT = Path("binance")
ZIPS = OUT / "zip"
CSVS = OUT / "csv"

ZIPS.mkdir(parents=True, exist_ok=True)
CSVS.mkdir(parents=True, exist_ok=True)

d = START
while d <= END:
    ds = d.isoformat()
    name = f"BTCUSD-2h-{ds}"
    url = f"{BASE}/{name}.zip"

    zip_path = ZIPS / f"{name}.zip"

    print(f"Downloading {name}...")
    if not zip_path.exists():
        urlretrieve(url, zip_path)

    with ZipFile(zip_path) as z:
        z.extractall(CSVS)

    d += timedelta(days=1)

# Compile columns:
# 2=Open, 3=High, 4=Low, 5=Close, 6=Volume, 9=Number of trades
output = OUT / "BTCUSD-2h-2025-12-04_to_2026-09-07.csv"

csv_files = sorted(CSVS.glob("BTCUSD-2h-*.csv"))

with output.open("w", newline="") as f_out:
    writer = csv.writer(f_out)

    # Header
    writer.writerow([
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades"
    ])

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}")

        with csv_file.open(newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or len(row) < 9:
                    continue

                writer.writerow([
                row[0],  # Open time
                row[1],  # Open
                row[2],  # High
                row[3],  # Low
                row[4],  # Close
                row[5],  # Volume
                row[8],  # Number of trades
            ])

print(f"\nDone: {output}")
