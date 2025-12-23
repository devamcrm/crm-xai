import polars as pl
from pathlib import Path

INPUT = "data/full_data.parquet"
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "all_features.parquet"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("📥 Loading full_data.parquet (lazy)...")
df = pl.scan_parquet(INPUT)

# --------------------------------------------------
# Purchases
# --------------------------------------------------
purchases = (
    df.filter(pl.col("event_type") == "purchase")
      .select([
          pl.col("source").alias("purchase_source"),
          pl.col("timestamp").alias("purchase_time"),
          pl.col("cat_0").alias("purchase_cat_0"),
          pl.col("user_id").alias("purchase_user_id"),
          pl.col("product_id").alias("purchase_pid"),
          pl.col("brand"),
          pl.col("price"),
      ])
      .with_columns(
          pl.concat_str(
              [
                  pl.col("purchase_user_id"),
                  pl.col("purchase_time").dt.strftime("%Y%m%d%H%M%S"),
              ],
              separator="_",
          ).alias("purchase_id")
      )
      .sort(["purchase_user_id", "purchase_time"])
)

# --------------------------------------------------
# Cart events
# --------------------------------------------------
carts = (
    df.filter(pl.col("event_type") == "cart")
      .select([
          pl.col("user_id"),
          pl.col("timestamp").alias("cart_time"),
          pl.col("product_id"),
          pl.col("cat_0"),
          pl.col("brand"),
          pl.col("price"),
      ])
)

# --------------------------------------------------
# cat_0 universe
# --------------------------------------------------
cat_0_values = (
    df.select("cat_0")
      .drop_nulls()
      .unique()
      .collect()
      .to_series()
      .to_list()
)

print(f"✅ Detected {len(cat_0_values)} cat_0 values")

# --------------------------------------------------
# Purchase history features
# --------------------------------------------------
print("🧮 Computing purchase history features...")

p = purchases.with_columns([
    pl.len().over("purchase_user_id").alias("purchase_idx"),
    pl.col("purchase_time").shift(1).over("purchase_user_id").alias("prev_purchase_time"),
])

p = p.with_columns([
    (pl.col("purchase_time") - pl.col("prev_purchase_time"))
        .dt.total_days()
        .alias("p_purchase_recency"),
    pl.col("purchase_idx").alias("p_purchase_count"),
])

# Average frequency (days between purchases)
p = p.with_columns(
    pl.when(pl.col("purchase_idx") > 1)
      .then(
          (pl.col("purchase_time") - pl.first("purchase_time").over("purchase_user_id"))
          .dt.total_days() / (pl.col("purchase_idx") - 1)
      )
      .otherwise(None)
      .alias("p_purchase_frequency")
)

# Cumulative purchase stats
p = p.with_columns([
    pl.col("price").cum_sum().shift(1).over("purchase_user_id").alias("p_purchase_value"),
    pl.col("purchase_pid").n_unique().over("purchase_user_id").shift(1).alias("p_purchase_products"),
    pl.col("purchase_cat_0").n_unique().over("purchase_user_id").shift(1).alias("p_purchase_cat_0"),
    pl.col("brand").n_unique().over("purchase_user_id").shift(1).alias("p_purchase_brands"),
])

# Per-category purchase counts
for c in cat_0_values:
    p = p.with_columns(
        pl.when(pl.col("purchase_cat_0") == c)
          .then(1)
          .otherwise(0)
          .cum_sum()
          .shift(1)
          .over("purchase_user_id")
          .alias(f"p_purchase_count_{c}")
    )

# --------------------------------------------------
# Cart history features
# --------------------------------------------------
print("🧺 Computing cart history features...")

pc = (
    p.join(carts, left_on="purchase_user_id", right_on="user_id", how="left")
     .filter(pl.col("cart_time") < pl.col("purchase_time"))
)

cart_aggs = pc.group_by("purchase_id").agg([

    pl.col("cart_time").max().alias("last_cart_time"),
    pl.col("cart_time").min().alias("first_cart_time"),

    pl.len().alias("cart_count"),
    pl.col("price").sum().alias("cart_value"),

    pl.col("product_id").n_unique().alias("cart_products"),
    pl.col("cat_0").n_unique().alias("cart_cat_0"),
    pl.col("brand").n_unique().alias("cart_brands"),

    *[
        (pl.col("cat_0") == c).sum().alias(f"cart_count_{c}")
        for c in cat_0_values
    ]
])

# Join purchase_time back for recency calc
cart_aggs = cart_aggs.join(
    p.select(["purchase_id", "purchase_time"]),
    on="purchase_id",
    how="left"
)

cart_aggs = cart_aggs.with_columns([

    (pl.col("purchase_time") - pl.col("last_cart_time"))
        .dt.total_days()
        .alias("cart_recency"),

    pl.when(pl.col("cart_count") > 1)
      .then(
          (pl.col("last_cart_time") - pl.col("first_cart_time"))
          .dt.total_days() / (pl.col("cart_count") - 1)
      )
      .otherwise(None)
      .alias("cart_frequency")
])

cart_aggs = cart_aggs.drop([
    "last_cart_time",
    "first_cart_time",
    "purchase_time",
])

# --------------------------------------------------
# Final assembly
# --------------------------------------------------
final = p.join(cart_aggs, on="purchase_id", how="left")

# --------------------------------------------------
# 🔑 CRITICAL FIX: cold-start + NULL handling
# --------------------------------------------------

# Explicit cold-start flag
final = final.with_columns(
    pl.when(pl.col("p_purchase_count") <= 1)
      .then(1)
      .otherwise(0)
      .alias("is_new_customer")
)

# Fill all numeric NULLs → 0 (LR-safe)
numeric_cols = [
    c for c, t in final.collect_schema().items()
    if t in (pl.Int64, pl.Int32, pl.UInt32, pl.Float64)
    and c != "purchase_cat_0"
]

final = final.with_columns([
    pl.col(c).fill_null(0).alias(c) for c in numeric_cols
])

# Target hygiene
final = final.with_columns(
    pl.col("purchase_cat_0")
      .fill_null("__UNKNOWN__")
      .alias("purchase_cat_0")
)

# --------------------------------------------------
# Output
# --------------------------------------------------
final_cols = [
    "purchase_source",
    "purchase_time",
    "purchase_cat_0",
    "purchase_user_id",
    "purchase_id",
    "is_new_customer",
    "p_purchase_recency",
    "p_purchase_frequency",
    "p_purchase_value",
    "p_purchase_count",
    "p_purchase_products",
    "p_purchase_cat_0",
    "p_purchase_brands",
] + [f"p_purchase_count_{c}" for c in cat_0_values] + [
    "cart_recency",
    "cart_frequency",
    "cart_value",
    "cart_count",
    "cart_products",
    "cart_cat_0",
    "cart_brands",
] + [f"cart_count_{c}" for c in cat_0_values] + [
    "purchase_pid"
]

print("💾 Writing all_features.parquet...")
final.select(final_cols).collect().write_parquet(OUT_PATH)

print("✅ 05_data_prepare.py completed successfully")
print("📌 Cold-start encoded | NULL-safe | LR + XGB ready")