import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

INPUT = "data/processed/all_features.parquet"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("📥 Loading all_features.parquet...")
df = pl.read_parquet(INPUT)

# --------------------------------------------------
# Identify numeric columns (exclude identifiers & target)
# --------------------------------------------------
EXCLUDE_COLS = {
    "purchase_source",
    "purchase_time",
    "purchase_cat_0",
    "purchase_user_id",
    "purchase_id",
    "purchase_pid",
}

NUM_COLS = [
    c for c, t in zip(df.columns, df.dtypes)
    if c not in EXCLUDE_COLS and t in (
        pl.Int64, pl.Int32, pl.UInt32, pl.Float64
    )
]

print(f"✅ Normalising {len(NUM_COLS)} numeric features")

# --------------------------------------------------
# Convert to pandas for sklearn scaler
# --------------------------------------------------
pdf = df.select(NUM_COLS).to_pandas()
pdf = pdf.fillna(0)

scaler = StandardScaler()
scaled = scaler.fit_transform(pdf)

# --------------------------------------------------
# Back to Polars (NO schema arg)
# --------------------------------------------------
scaled_df = pl.DataFrame(
    scaled,
    schema={c: pl.Float64 for c in NUM_COLS}
)

# --------------------------------------------------
# Reattach non-numeric columns
# --------------------------------------------------
final = pl.concat(
    [
        df.select([c for c in df.columns if c not in NUM_COLS]),
        scaled_df
    ],
    how="horizontal"
)

# --------------------------------------------------
# Write normalized dataset
# --------------------------------------------------
OUT_N = OUT_DIR / "all_features_n.parquet"
final.write_parquet(OUT_N)

print(f"💾 Normalized file written: {OUT_N}")

# --------------------------------------------------
# Split normalized data
# --------------------------------------------------
print("✂️ Splitting normalized dataset...")

final.filter(pl.col("purchase_source") == "train") \
     .write_parquet(OUT_DIR / "all_features_train_n.parquet")

final.filter(pl.col("purchase_source") == "val") \
     .write_parquet(OUT_DIR / "all_features_val_n.parquet")

final.filter(pl.col("purchase_source") == "test") \
     .write_parquet(OUT_DIR / "all_features_test_n.parquet")

# --------------------------------------------------
# Save scaler for inference
# --------------------------------------------------
joblib.dump(scaler, OUT_DIR / "feature_scaler.pkl")

print("✅ Normalization + split completed successfully")
print("📦 Outputs:")
print("- all_features_n.parquet")
print("- all_features_train_n.parquet")
print("- all_features_val_n.parquet")
print("- all_features_test_n.parquet")
print("- feature_scaler.pkl")
