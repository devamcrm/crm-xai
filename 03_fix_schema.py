import polars as pl

INPUT_PATH = "data/full_data.parquet"
OUTPUT_PATH = "data/full_data.parquet"  # overwrite intentionally

print("🧹 Fixing schema for full_data.parquet\n")

# Lazy load
lf = pl.scan_parquet(INPUT_PATH)

print("🔎 Original schema:")
for k, v in lf.collect_schema().items():
    print(f"{k:20s} → {v}")

# Apply schema corrections
lf_fixed = lf.with_columns([
    # event_time: String → Datetime UTC
    pl.col("event_time")
      .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False)
      .dt.replace_time_zone("UTC")
      .alias("event_time"),

    # timestamp: enforce UTC timezone
    pl.col("timestamp")
      .dt.replace_time_zone("UTC")
      .alias("timestamp"),

    # price: String → Float64
    pl.col("price")
      .cast(pl.Float64, strict=False)
      .alias("price"),
])

# Write back (streaming-safe)
lf_fixed.collect(streaming=True).write_parquet(OUTPUT_PATH)

print("\n✅ Schema fix complete.\n")

# Recheck schema
lf_check = pl.scan_parquet(OUTPUT_PATH)
print("🔎 Fixed schema:")
for k, v in lf_check.collect_schema().items():
    print(f"{k:20s} → {v}")
