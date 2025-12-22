import polars as pl
from pathlib import Path

DATA_PATH = Path("data/full_data.parquet")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/full_data.parquet not found. Run data_combine.py first."
    )

print("🧹 Loading full_data.parquet...")
lf = pl.scan_parquet(DATA_PATH)

schema = lf.collect_schema()
print("📐 Original columns:", list(schema.keys()))

# Columns where the literal value "NA" should be treated as missing data
NA_TO_NULL_COLS = ["brand", "cat_0", "cat_1", "cat_2", "cat_3"]

clean_exprs = []
for col in NA_TO_NULL_COLS:
    if col in schema:
        clean_exprs.append(
            pl.when(pl.col(col) == "NA")
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

# Apply NA → null normalization
lf_clean = lf.with_columns(clean_exprs)

# Drop event_time due to duplication with the canonical timestamp column
if "event_time" in schema:
    lf_clean = lf_clean.drop("event_time")
    print("🗑 Dropped column: event_time")

print("💾 Writing cleaned full_data.parquet...")
lf_clean.collect(engine="streaming").write_parquet(DATA_PATH)

print("✅ Data cleanup complete.")
