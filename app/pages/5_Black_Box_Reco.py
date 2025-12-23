import streamlit as st
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

# ---------- Page Config ----------
st.set_page_config(
    page_title="Black-Box Recommendation Engine",
    layout="wide"
)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "demo"
MODEL_DIR = BASE_DIR / "models" / "reco"

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgboost.pkl")

    return feature_cols, label_encoder, xgb_model


feature_cols, label_encoder, xgb_model = load_models()

# ---------- Load Demo Data ----------
@st.cache_data
def load_demo_data():
    events = pd.read_parquet(DATA_DIR / "demo_user_events.parquet")
    features = pd.read_parquet(DATA_DIR / "demo_user_features.parquet")
    return events, features


events_df, features_df = load_demo_data()

# ---------- Helpers ----------
def prepare_X_for_model(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Align demo features with model-required features.
    Missing features (e.g. is_new_customer) are injected as 0.
    """
    X = df.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


def predict_top_k(model, X, k=3):
    probs = model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [(label_encoder.classes_[i], float(probs[i])) for i in top_idx]


# ---------- UI ----------
st.title("Black-Box Recommendation Engine")

st.markdown("""
This page demonstrates a **black-box recommendation experience**.

You can see **what** the model recommends, but **not why**.
This mirrors how many real-world AI systems are presented to business users.
""")

st.markdown("---")

# ---------- User Selection ----------
user_ids = sorted(features_df["user_id"].unique())
selected_user = st.selectbox("Select a demo user", user_ids)

user_features = features_df[features_df["user_id"] == selected_user]
user_events = events_df[events_df["user_id"] == selected_user]

# ---------- Feature Alignment ----------
X = prepare_X_for_model(user_features, feature_cols)

# ---------- Predictions (XGBoost only) ----------
xgb_top = predict_top_k(xgb_model, X, k=3)

# ---------- Display Predictions ----------
st.header("Recommended Categories")

for rank, (cat, prob) in enumerate(xgb_top, start=1):
    st.write(f"**{rank}. {cat}** — {prob:.2%}")

st.markdown("""
The model internally evaluates multiple categories, but only confidence scores
are shown here — **no explanations or feature-level reasoning is provided**.
""")

st.markdown("---")

# ---------- Event Log ----------
st.header("User Event Log")
st.dataframe(
    user_events.sort_values("timestamp", ascending=False),
    use_container_width=True
)

# ---------- Feature Values ----------
st.header("Model Input Features")
st.dataframe(
    X.T.rename(columns={X.index[0]: "value"}),
    use_container_width=True
)

st.caption(
    "Missing features (e.g. cold-start flags) are injected as zero at inference time. "
    "This is standard practice in production recommendation systems."
)
