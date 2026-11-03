
from pathlib import Path
import duckdb

data_folder = Path("data")

files = [
    file for file in data_folder.rglob("*.parquet")
    if not file.name.endswith("_full_analysis.parquet")
]

print("Original Parquet files found:", len(files))

if not files:
    print("No original Parquet files found.")
else:
    file = files[0]
    print("\nInspecting:", file)

    con = duckdb.connect()

    schema = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(file)]
    ).fetchdf()

    print("\nDATASET SCHEMA:")
    print(schema.to_string(index=False))

    print("\nSAMPLE RECORD:")
    sample = con.execute(
        "SELECT * FROM read_parquet(?) LIMIT 1",
        [str(file)]
    ).fetchdf()

    print(sample.to_string())

    con.close()
