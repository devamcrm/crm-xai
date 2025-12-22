# 10_reco_engine.py
# -------------------------------------------------------
# Multiclass Recommendation Models for purchase_cat_0
# Models:
# 1. Logistic Regression
# 2. XGBoost
# -------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import joblib


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/reco")

TRAIN_PATH = DATA_DIR / "all_features_train_n.parquet"
VAL_PATH   = DATA_DIR / "all_features_val_n.parquet"
TEST_PATH  = DATA_DIR / "all_features_test_n.parquet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

print("📥 Loading datasets...")

train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
test_df  = pd.read_parquet(TEST_PATH)

print(f"Train rows: {len(train_df):,}")
print(f"Val rows:   {len(val_df):,}")
print(f"Test rows:  {len(test_df):,}")


# -------------------------------------------------------
# Target preprocessing
# -------------------------------------------------------

TARGET_COL = "purchase_cat_0"
UNKNOWN_LABEL = "__UNKNOWN__"

for df in (train_df, val_df, test_df):
    df[TARGET_COL] = df[TARGET_COL].fillna(UNKNOWN_LABEL).astype(str)

y_train_raw = train_df[TARGET_COL]
y_val_raw   = val_df[TARGET_COL]
y_test_raw  = test_df[TARGET_COL]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_val   = label_encoder.transform(y_val_raw)
y_test  = label_encoder.transform(y_test_raw)

joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")


# -------------------------------------------------------
# Feature selection
# -------------------------------------------------------

BASE_FEATURES = [
    "p_purchase_recency",
    "p_purchase_frequency",
    "p_purchase_value",
    "p_purchase_count",
    "p_purchase_products",
    "p_purchase_cat_0",
    "p_purchase_brands",
    "cart_recency",
    "cart_value",
    "cart_frequency",
    "cart_count",
    "cart_products",
    "cart_cat_0",
    "cart_brands",
]

P_PURCHASE_CAT_FEATURES = sorted(
    c for c in train_df.columns if c.startswith("p_purchase_count_")
)

CART_CAT_FEATURES = sorted(
    c for c in train_df.columns if c.startswith("cart_count_")
)

FEATURE_COLS = BASE_FEATURES + P_PURCHASE_CAT_FEATURES + CART_CAT_FEATURES

print(f"✅ Using {len(FEATURE_COLS)} features")

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)


# -------------------------------------------------------
# Feature matrices
# -------------------------------------------------------

X_train = train_df[FEATURE_COLS].fillna(0)
X_val   = val_df[FEATURE_COLS].fillna(0)
X_test  = test_df[FEATURE_COLS].fillna(0)


# -------------------------------------------------------
# Evaluation helper
# -------------------------------------------------------

def evaluate(model_name, model, X, y_true):
    preds = model.predict(X)
    acc = accuracy_score(y_true, preds)
    print(f"\n📊 {model_name} Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true,
            preds,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


# -------------------------------------------------------
# Logistic Regression
# -------------------------------------------------------

print("\n🧠 Training Logistic Regression model...")

lr_model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)

lr_model.fit(X_train, y_train)

joblib.dump(lr_model, MODEL_DIR / "logistic_regression.pkl")

print("\n🔎 Logistic Regression – Validation")
evaluate("Logistic Regression (Val)", lr_model, X_val, y_val)

print("\n🔎 Logistic Regression – Test")
evaluate("Logistic Regression (Test)", lr_model, X_test, y_test)


# -------------------------------------------------------
# XGBoost
# -------------------------------------------------------

print("\n🧠 Training XGBoost model...")

xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

joblib.dump(xgb_model, MODEL_DIR / "xgboost.pkl")

print("\n🔎 XGBoost – Validation")
evaluate("XGBoost (Val)", xgb_model, X_val, y_val)

print("\n🔎 XGBoost – Test")
evaluate("XGBoost (Test)", xgb_model, X_test, y_test)


# -------------------------------------------------------
# Metadata
# -------------------------------------------------------

meta = {
    "models": [
        "LogisticRegression",
        "XGBoostClassifier"
    ],
    "n_features": len(FEATURE_COLS),
    "target": TARGET_COL,
    "classes": list(label_encoder.classes_),
    "unknown_class_label": UNKNOWN_LABEL,
}

with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Recommendation model training completed")
print(f"📦 Model artefacts saved in: {MODEL_DIR}")