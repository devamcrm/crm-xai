import polars as pl

PATH = "data/processed/all_features.parquet"

print("\n📦 Loading dataset (lazy)...")
df = pl.scan_parquet(PATH)

# -----------------------------
# 1. Table overview
# -----------------------------
schema = df.collect_schema()
cols = schema.names()

row_count = df.select(pl.len()).collect().item()

print("\n📊 Table overview")
print(f"Rows: {row_count:,}")
print(f"Columns: {len(cols)}")

# -----------------------------
# 2. Critical columns
# -----------------------------
required_cols = {
    "purchase_source",
    "purchase_time",
    "purchase_cat_0",
    "purchase_user_id",
}

missing = required_cols - set(cols)
assert not missing, f"❌ Missing required columns: {missing}"
print("✅ Critical columns present")

# -----------------------------
# 3. Purchase-level uniqueness diagnostics
# -----------------------------
print("\n🔍 Purchase-level uniqueness check")

unique_purchases = (
    df.select(["purchase_user_id", "purchase_time"])
      .unique()
      .select(pl.len())
      .collect()
      .item()
)

dup_rows = row_count - unique_purchases
dup_factor = row_count / unique_purchases

print(f"Unique purchases: {unique_purchases:,}")
print(f"Duplicate rows:   {dup_rows:,}")
print(f"Duplication x:    {dup_factor:.2f}")

# -----------------------------
# 4. Sample duplicated purchases
# -----------------------------
print("\n🧪 Sample duplicated purchase keys")

dupes = (
    df.group_by(["purchase_user_id", "purchase_time"])
      .len()
      .filter(pl.col("len") > 1)
      .sort("len", descending=True)
      .limit(10)
      .collect()
)

if dupes.height == 0:
    print("✅ No duplicated purchases")
else:
    print(dupes)

# -----------------------------
# 5. Column diagnostics
# -----------------------------
print("\n🧾 Column diagnostics")
print("-" * 90)
print(f"{'Column':30} {'Type':25} {'Distinct':>10}")
print("-" * 90)

for col in cols:
    dtype = schema[col]
    distinct = (
        df.select(pl.col(col).n_unique())
          .collect()
          .item()
    )
    print(f"{col:30} {str(dtype):25} {distinct:>10}")

# -----------------------------
# 6. Top 5 values per column
# -----------------------------
print("\n🔎 Top 5 values per column (by frequency)\n")

for col in cols:
    print(f"▶ {col}")
    try:
        top_vals = (
            df.select(col)
              .group_by(col)
              .len()
              .sort("len", descending=True)
              .limit(5)
              .collect()
        )
        for row in top_vals.iter_rows():
            print(f"  {row[0]} → {row[1]:,}")
    except Exception as e:
        print(f"  ⚠️ {e}")
    print()

print("✅ Feature sanity diagnostics completed")
