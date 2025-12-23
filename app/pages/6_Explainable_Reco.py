import streamlit as st
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable Recommendation Engine (XAI)",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "demo"
MODEL_DIR = BASE_DIR / "models" / "reco"

# --------------------------------------------------
# Load model + metadata
# --------------------------------------------------
@st.cache_resource
def load_model_assets():
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgboost.pkl")

    return feature_cols, label_encoder, xgb_model


feature_cols, label_encoder, xgb_model = load_model_assets()

# --------------------------------------------------
# Load demo data
# --------------------------------------------------
@st.cache_data
def load_demo_data():
    events = pd.read_parquet(DATA_DIR / "demo_user_events.parquet")
    features = pd.read_parquet(DATA_DIR / "demo_user_features.parquet")
    return events, features


events_df, features_df = load_demo_data()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Add missing model features safely (e.g. is_new_customer)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    return X


def get_top_predictions(model, X, k=2):
    probs = model.predict_proba(X)[0]
    idx = np.argsort(probs)[::-1][:k]
    return [(label_encoder.classes_[i], probs[i]) for i in idx]


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Explainable Recommendation Engine (XAI)")

st.markdown("""
This page demonstrates an **explainable AI recommendation system** using **SHAP**.

Unlike the black-box view, this page shows:
- *Why* a category was recommended
- *Which features influenced the decision*
- *Why an alternative category was not chosen*
""")

st.markdown("---")

# --------------------------------------------------
# User selection
# --------------------------------------------------
user_ids = sorted(features_df["user_id"].unique())
selected_user = st.selectbox("Select a demo user", user_ids)

user_features = features_df[features_df["user_id"] == selected_user]
user_events = events_df[events_df["user_id"] == selected_user]

X = prepare_X(user_features)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
top_preds = get_top_predictions(xgb_model, X, k=2)

(top1_label, top1_prob), (top2_label, top2_prob) = top_preds

st.subheader("Model Predictions")

st.write(f"**Top Recommendation:** {top1_label} — {top1_prob:.2%}")
st.write(f"**Next Best Alternative:** {top2_label} — {top2_prob:.2%}")

st.markdown("---")

# --------------------------------------------------
# SHAP Explainer (TreeExplainer – correct usage)
# --------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

top1_class_idx = label_encoder.transform([top1_label])[0]

# --------------------------------------------------
# Handle SHAP output formats (critical fix)
# --------------------------------------------------

if isinstance(shap_values, list):
    # Legacy multiclass format: list[class] -> (n_samples, n_features)
    shap_local = shap_values[top1_class_idx][0]

elif isinstance(shap_values, np.ndarray):
    # New format: (n_samples, n_features, n_classes)
    shap_local = shap_values[0, :, top1_class_idx]

else:
    raise RuntimeError("Unsupported SHAP output format")

# --------------------------------------------------
# SHAP feature alignment (robust for XGBoost)
# --------------------------------------------------

n_shap_features = len(shap_local)

# SHAP may not expose feature names → fall back to X columns
if explainer.feature_names is not None:
    shap_feature_names = explainer.feature_names
else:
    shap_feature_names = X.columns[:n_shap_features].tolist()

# Final safety check
assert len(shap_feature_names) == n_shap_features, (
    f"SHAP feature mismatch: "
    f"{len(shap_feature_names)} names vs {n_shap_features} values"
)

# Extract corresponding feature values
X_row = X.iloc[0][shap_feature_names].values

shap_df = (
    pd.DataFrame({
        "Feature": shap_feature_names,
        "SHAP Value": shap_local,
        "Feature Value": X_row
    })
    .assign(abs_val=lambda d: d["SHAP Value"].abs())
    .sort_values("abs_val", ascending=False)
    .drop(columns="abs_val")
)

# --------------------------------------------------
# Local explanation (Top-1 decision)
# --------------------------------------------------
st.subheader("Why this category was recommended")

st.markdown(f"""
The chart below shows the **most influential features** that led the model to recommend  
**{top1_label}** over other categories.

- Positive values → pushed the prediction *towards* {top1_label}
- Negative values → pushed the prediction *away*
""")

TOP_N = 12
local_top = shap_df.head(TOP_N)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    local_top["Feature"][::-1],
    local_top["SHAP Value"][::-1]
)
ax.set_title("Local Feature Contributions (SHAP)")
ax.set_xlabel("Impact on Model Output")
st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# Top-1 vs Top-2 comparison (probability logic)
# --------------------------------------------------
st.subheader("Why the alternative category was not chosen")

st.markdown(f"""
Although **{top2_label}** was a strong alternative, the model assigned a **higher overall
confidence** to **{top1_label}**.

- Probability difference: **{(top1_prob - top2_prob):.2%}**
- The features above contributed more strongly toward **{top1_label}**
""")

# --------------------------------------------------
# Global explanation (dataset-level)
# --------------------------------------------------
st.subheader("Global Feature Importance (Model-level)")

st.markdown("""
This chart shows which features the model relies on **most often across all users**.
""")

# Use XGBoost gain importance
importance = xgb_model.get_booster().get_score(importance_type="gain")
global_df = (
    pd.DataFrame({
        "Feature": importance.keys(),
        "Importance": importance.values()
    })
    .sort_values("Importance", ascending=False)
    .head(15)
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(
    global_df["Feature"][::-1],
    global_df["Importance"][::-1]
)
ax2.set_title("Global Feature Importance (XGBoost)")
ax2.set_xlabel("Average Gain")
st.pyplot(fig2)

st.markdown("---")

# --------------------------------------------------
# Event log
# --------------------------------------------------
st.subheader("User Event Log")
st.dataframe(
    user_events.sort_values("timestamp", ascending=False),
    use_container_width=True
)

# --------------------------------------------------
# Feature values
# --------------------------------------------------
st.subheader("Model Input Features")
st.dataframe(
    X.T.rename(columns={X.index[0]: "value"}),
    use_container_width=True
)

st.caption(
    "SHAP explanations are shown for the final decision made by the model. "
    "Alternative categories are compared using predicted probabilities."
)