import polars as pl

DATA_PATH = "data/full_data.parquet"

print(f"\n🔍 Inspecting schema of {DATA_PATH}\n")

lf = pl.scan_parquet(DATA_PATH)

schema = lf.collect_schema()

print("Column → DataType | Distinct values")
print("-" * 55)

distinct_counts = (
    lf.select([
        pl.col(c).n_unique().alias(c)
        for c in schema.keys()
    ])
    .collect()
    .to_dict(as_series=False)
)

for col, dtype in schema.items():
    n_unique = distinct_counts[col][0]
    print(f"{col:<20} → {str(dtype):<25} | {n_unique}")

print("\n🔎 Top 5 values per column (by frequency)\n")

for col in schema.keys():
    print(f"▶ Column: {col}")

    top_vals = (
        lf
        .group_by(col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(5)
        .collect()
    )

    for row in top_vals.iter_rows():
        print(f"  {row[0]} → {row[1]}")

    print()
