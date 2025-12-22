import polars as pl
from pathlib import Path

INPUT_PATH = "data/processed/all_features.parquet"
OUT_DIR = Path("data/processed")

TRAIN_PATH = OUT_DIR / "all_features_train.parquet"
VAL_PATH   = OUT_DIR / "all_features_val.parquet"
TEST_PATH  = OUT_DIR / "all_features_test.parquet"

print("📥 Loading all_features.parquet (lazy)...")
df = pl.scan_parquet(INPUT_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("✂️ Splitting dataset by purchase_source...")

df.filter(pl.col("purchase_source") == "train") \
  .collect() \
  .write_parquet(TRAIN_PATH)

df.filter(pl.col("purchase_source") == "val") \
  .collect() \
  .write_parquet(VAL_PATH)

df.filter(pl.col("purchase_source") == "test") \
  .collect() \
  .write_parquet(TEST_PATH)

print("✅ Split completed")

print("\n📊 Row counts:")
print(f"Train: {pl.read_parquet(TRAIN_PATH).height:,}")
print(f"Val:   {pl.read_parquet(VAL_PATH).height:,}")
print(f"Test:  {pl.read_parquet(TEST_PATH).height:,}")

print("\n📦 Output files:")
print(f"- {TRAIN_PATH}")
print(f"- {VAL_PATH}")
print(f"- {TEST_PATH}")
