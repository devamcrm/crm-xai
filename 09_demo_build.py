import polars as pl
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------
# Config
# -----------------------------
FULL_DATA = "data/full_data.parquet"
OUT_EVENTS = "data/demo/demo_user_events.parquet"
OUT_FEATURES = "data/demo/demo_user_features.parquet"

N_USERS = 10_000
PREDICTION_TIME = datetime(2020, 5, 5, 0, 0, 0, tzinfo=timezone.utc)

Path("data/demo").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load full dataset
# -----------------------------
print("📥 Loading full_data.parquet (lazy)...")
df = pl.scan_parquet(FULL_DATA)

# -----------------------------
# Step 1: users with ≥1 TEST purchase
# -----------------------------
print("🔍 Selecting users with ≥1 TEST purchase...")

eligible_users = (
    df.filter(
        (pl.col("source") == "test") &
        (pl.col("event_type") == "purchase")
    )
    .select("user_id")
    .unique()
    .collect()
)

print(f"✅ Eligible users found: {eligible_users.height}")

sampled_users = eligible_users.sample(
    n=min(N_USERS, eligible_users.height),
    seed=42
)

print(f"🎯 Sampled users: {sampled_users.height}")

sampled_users_lf = pl.LazyFrame(sampled_users)

# -----------------------------
# Step 2–3: extract all events
# -----------------------------
print("📦 Extracting all events for sampled users...")

demo_events = (
    df.join(sampled_users_lf, on="user_id", how="inner")
    .collect()
)

demo_events.write_parquet(OUT_EVENTS)

print("✅ demo_user_events.parquet written")
print(f"   Rows: {demo_events.height}")

# -----------------------------
# Step 4: build one-row-per-user features
# -----------------------------
print("📊 Building one-row-per-user feature table...")

cat_0_values = (
    demo_events
    .select("cat_0")
    .drop_nulls()
    .unique()
    .to_series()
    .to_list()
)

# Split purchase & cart
purchases = demo_events.filter(pl.col("event_type") == "purchase")
carts = demo_events.filter(pl.col("event_type") == "cart")

# -----------------------------
# Purchase aggregates
# -----------------------------
purchase_aggs = [
    pl.lit(None).alias("purchase_source"),
    pl.lit(PREDICTION_TIME).alias("purchase_time"),
    pl.lit(None).alias("purchase_cat_0"),
    pl.col("user_id").alias("purchase_user_id"),
    (
        pl.col("user_id") +
        "_" +
        pl.lit(PREDICTION_TIME.strftime("%Y%m%d%H%M%S"))
    ).alias("purchase_id"),

    pl.lit(0).cast(pl.Int32).alias("is_new_customer"),

    # Recency
    (pl.lit(PREDICTION_TIME) - pl.col("timestamp").max())
        .dt.total_days()
        .cast(pl.Int64)
        .alias("p_purchase_recency"),

    # Frequency
    pl.when(pl.count() > 1)
      .then(
          pl.col("timestamp")
            .sort()
            .diff()
            .dt.total_days()
            .mean()
      )
      .otherwise(None)
      .alias("p_purchase_frequency"),

    pl.sum("price").alias("p_purchase_value"),
    pl.count().alias("p_purchase_count"),
    pl.col("product_id").n_unique().alias("p_purchase_products"),
    pl.col("cat_0").n_unique().alias("p_purchase_cat_0"),
    pl.col("brand").n_unique().alias("p_purchase_brands"),
]

for c in cat_0_values:
    purchase_aggs.append(
        pl.when(pl.col("cat_0") == c)
          .then(1)
          .otherwise(0)
          .sum()
          .alias(f"p_purchase_count_{c}")
    )

purchase_features = (
    purchases
    .group_by("user_id")
    .agg(purchase_aggs)
)

# -----------------------------
# Cart aggregates
# -----------------------------
cart_aggs = [
    (pl.lit(PREDICTION_TIME) - pl.col("timestamp").max())
        .dt.total_days()
        .cast(pl.Int64)
        .alias("cart_recency"),

    pl.sum("price").alias("cart_value"),

    pl.when(pl.count() > 1)
      .then(
          pl.col("timestamp")
            .sort()
            .diff()
            .dt.total_days()
            .mean()
      )
      .otherwise(None)
      .alias("cart_frequency"),

    pl.count().alias("cart_count"),
    pl.col("product_id").n_unique().alias("cart_products"),
    pl.col("cat_0").n_unique().alias("cart_cat_0"),
    pl.col("brand").n_unique().alias("cart_brands"),
]

for c in cat_0_values:
    cart_aggs.append(
        pl.when(pl.col("cat_0") == c)
          .then(1)
          .otherwise(0)
          .sum()
          .alias(f"cart_count_{c}")
    )

cart_features = (
    carts
    .group_by("user_id")
    .agg(cart_aggs)
)

# -----------------------------
# Final join
# -----------------------------
final_features = (
    purchase_features
    .join(cart_features, on="user_id", how="left")
    .with_columns(
        pl.lit(None).alias("purchase_pid")
    )
)

final_features.write_parquet(OUT_FEATURES)

print("✅ demo_user_features.parquet written")
print(f"   Rows: {final_features.height}")
print("✅ Demo build completed successfully")
