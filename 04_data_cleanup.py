import polars as pl
from pathlib import Path

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_PATH = Path("data/full_data.parquet")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "❌ data/full_data.parquet not found. Run 01_data_combine.py first."
    )

print("🧹 Loading full_data.parquet (lazy scan)...")
lf = pl.scan_parquet(DATA_PATH)

schema = lf.collect_schema()
print(f"📐 Columns detected: {len(schema)}")

# --------------------------------------------------
# 1. Normalise literal 'NA' strings → NULL
# --------------------------------------------------
NA_TO_NULL_COLS = ["brand", "cat_0", "cat_1", "cat_2", "cat_3", "purchase_cat_0"]

clean_exprs = []
for col in NA_TO_NULL_COLS:
    if col in schema:
        clean_exprs.append(
            pl.when(pl.col(col) == "NA")
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

lf = lf.with_columns(clean_exprs)
print(f"🔁 Normalised 'NA' → NULL for {len(clean_exprs)} columns")

# --------------------------------------------------
# 2. Drop duplicated / unused timestamp column
# --------------------------------------------------
if "event_time" in schema:
    lf = lf.drop("event_time")
    print("🗑 Dropped column: event_time")

# --------------------------------------------------
# 3. CART FEATURES
# NULL = no cart activity → safe to impute 0
# --------------------------------------------------
cart_cols = [c for c in schema if c.startswith("cart_")]

cart_exprs = []
for col in cart_cols:
    dtype = schema[col]

    if dtype in (pl.Int64, pl.Int32, pl.UInt32):
        cart_exprs.append(pl.col(col).fill_null(0).alias(col))
    elif dtype == pl.Float64:
        cart_exprs.append(pl.col(col).fill_null(0.0).alias(col))

lf = lf.with_columns(cart_exprs)
print(f"🛒 Imputed NULL → 0 for {len(cart_exprs)} cart features")

# --------------------------------------------------
# 4. PURCHASE FEATURES (CRITICAL FIX)
# NULL = no purchase history (cold-start)
# --------------------------------------------------

# 4a. Add explicit cold-start indicator
if "p_purchase_recency" in schema:
    lf = lf.with_columns(
        pl.when(pl.col("p_purchase_recency").is_null())
        .then(1)
        .otherwise(0)
        .alias("is_new_customer")
    )
    print("🆕 Added cold-start flag: is_new_customer")

# 4b. Impute purchase NULLs → 0 (after flag)
purchase_cols = [
    c for c in schema
    if c.startswith("p_purchase_") and c != "p_purchase_recency"
]

purchase_exprs = []
for col in purchase_cols:
    dtype = schema[col]

    if dtype in (pl.Int64, pl.Int32, pl.UInt32):
        purchase_exprs.append(pl.col(col).fill_null(0).alias(col))
    elif dtype == pl.Float64:
        purchase_exprs.append(pl.col(col).fill_null(0.0).alias(col))

# Also fill recency AFTER flag creation
if "p_purchase_recency" in schema:
    purchase_exprs.append(
        pl.col("p_purchase_recency").fill_null(0).alias("p_purchase_recency")
    )

lf = lf.with_columns(purchase_exprs)
print(f"🧯 Imputed NULL → 0 for {len(purchase_exprs)} purchase features")

# --------------------------------------------------
# 5. Target hygiene
# Keep UNKNOWN but normalise nulls
# --------------------------------------------------
if "purchase_cat_0" in schema:
    lf = lf.with_columns(
        pl.col("purchase_cat_0")
        .fill_null("__UNKNOWN__")
        .alias("purchase_cat_0")
    )
    print("🎯 Normalised target NULL → '__UNKNOWN__'")

# --------------------------------------------------
# 6. Final write
# --------------------------------------------------
print("💾 Writing cleaned full_data.parquet...")
lf.collect(engine="streaming").write_parquet(DATA_PATH)

print("✅ Data cleanup complete.")
print("📌 Cold-start logic preserved | LR-safe | XGB-safe")