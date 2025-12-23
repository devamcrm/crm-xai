import streamlit as st
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(
    page_title="About Models & Results",
    layout="wide"
)

# ---------- Logo ----------
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "uol_logo.svg"

st.image(str(LOGO_PATH), width=160)
st.markdown("---")

# ---------- Title ----------
st.title("About the Models and Results")

st.markdown("""
This page explains the **machine learning models** used in this study and how
their outputs should be interpreted within the context of this research.
""")

# ---------- Models Used ----------
st.header("Models used in this study")

st.markdown("""
Two different machine learning models are used in this prototype recommendation system:

### 1. Logistic Regression
Logistic Regression is used as a **baseline model**.

- It is a well-established, linear classification technique
- It is relatively simple and transparent by design
- Feature contributions are easier to interpret
- It represents how many real-world CRM systems operate today

This model helps establish a reference point for understanding recommendations
with minimal complexity.
""")

st.markdown("""
### 2. XGBoost Classifier
XGBoost is a more **powerful, non-linear model** based on gradient-boosted decision trees.

- It can capture complex interactions between features
- It typically achieves stronger predictive performance than linear models
- Its internal decision logic is not directly interpretable
- It represents modern, production-grade AI systems used in large-scale CRM platforms

In this study, XGBoost is treated as a **black-box model** unless explanations are added.
""")

# ---------- Input & Target ----------
st.header("What the models predict")

st.markdown("""
Both models are trained to predict the **next likely purchase category** for a customer
based on historical behaviour.

Key characteristics:

- **Number of input features:** 40 engineered behavioural and transactional features
- **Prediction target:** Next purchase category (`purchase_cat_0`)
- **Problem type:** Multi-class classification
""")

st.markdown("""
The possible output categories include common retail segments such as:

- Apparel
- Electronics
- Furniture
- Accessories
- Sports
- Kids
- Medicine
- Stationery  
…and others

An additional `__UNKNOWN__` category is used to handle edge cases and sparse histories.
""")

# ---------- About Results ----------
st.header("How to interpret the results")

st.markdown("""
It is important to note that this research **does not aim to optimise or benchmark model accuracy**.

Instead, the focus is on:

- How recommendations are **perceived by business users**
- Whether explanations increase **trust and confidence**
- How transparency affects **willingness to act on AI recommendations**
""")

st.markdown("""
As a result:

- Performance metrics such as accuracy or F1-score are **not emphasised**
- Recommendations shown are **illustrative**, not prescriptive
- Both models use the **same underlying data and features**
""")

# ---------- Explainability ----------
st.header("Explainability and SHAP")

st.markdown("""
To introduce transparency, this study uses **SHAP (SHapley Additive exPlanations)**.

SHAP is an explainability technique that:

- Explains individual predictions after the model has made them
- Shows which features contributed positively or negatively
- Works with complex models such as XGBoost
""")

st.markdown("""
Importantly, SHAP:

- Does **not** change the model’s predictions
- Acts as an **interpretation layer**
- Allows business users to understand *why* a recommendation was generated
""")

# ---------- What comes next ----------
st.header("What you will see next")

st.markdown("""
In the next sections of this app, you will experience:

- A **black-box recommendation**, where no explanation is provided
- An **explainable recommendation**, where SHAP-based insights are shown

You will then be asked to reflect on differences in:
- Trust
- Transparency
- Usability
- Willingness to rely on the recommendation in a real CRM setting
""")

st.info(
    "There are no right or wrong answers. The study focuses on your professional judgement and perception."
)
