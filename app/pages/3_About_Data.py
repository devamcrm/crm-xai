import streamlit as st
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(
    page_title="About the Data",
    layout="wide"
)

# ---------- Logo ----------
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "uol_logo.svg"

st.image(str(LOGO_PATH), width=160)
st.markdown("---")

# ---------- Page Title ----------
st.title("About the Data Used in This Study")

st.markdown("""
This page explains **what type of data is used**, **where it comes from**,  
and **how it has been prepared** for the purposes of this research study.
""")

# ---------- Data overview ----------
st.header("Overview of the data")

st.markdown("""
The recommendation engine demonstrated in this application is based on
**publicly available, anonymised ecommerce interaction data**
that reflects typical information used in
Customer Relationship Management (CRM) and recommendation systems.

No real customers can be identified from the data shown in this app.
""")

# ---------- Data source ----------
st.header("Data source")

st.markdown("""
The original dataset used as the basis for this study was obtained from **Kaggle**,
a widely used platform for sharing open datasets for data science and machine learning research.

Specifically, the data originates from the following Kaggle dataset:
""")

st.markdown("""
🔗 **RecSys 2020 E-commerce Dataset**  
https://www.kaggle.com/datasets/dschettler8845/recsys-2020-ecommerce-dataset/data
""")

st.markdown("""
This dataset was created for research and experimentation in recommendation systems
and represents **anonymised ecommerce user interactions and events**.
""")

# ---------- Adaptation for this study ----------
st.header("How the data is used in this study")

st.markdown("""
The Kaggle dataset has **not been used in its raw form**.
Instead, it has been **processed, filtered, and adapted** specifically for this MSc research project.

Key adaptations include:
- Removal of any remaining identifiers
- Aggregation and transformation into CRM-style features
- Simplification for interpretability and explainability analysis
""")

# ---------- Why this data ----------
st.header("Why this type of data is appropriate")

st.markdown("""
Modern CRM and recommendation systems commonly rely on behavioural and transactional data
to predict what a customer might be interested in next.

To reflect realistic business scenarios, the adapted dataset represents patterns such as:
- Historical interaction frequency
- Product or category engagement
- Recency of user activity
- Behavioural consistency over time

These data characteristics closely mirror those used in real-world CRM platforms.
""")

# ---------- Anonymisation ----------
st.header("Anonymisation and ethical use")

st.markdown("""
The data used in this application should be considered **fully anonymised and research-adapted**.

This means:
- No real customer identities are present
- No personal or sensitive information is used
- The data cannot be traced back to individuals

The purpose of using this dataset is **not** to analyse customers,
but to study **how business users perceive AI recommendations and explanations**.
""")

# ---------- Feature examples ----------
st.header("Examples of data features")

st.markdown("""
While values have been adapted for research purposes,
the data structure reflects common CRM and recommendation features such as:

- **Engagement frequency**
- **Recency of interaction**
- **Category affinity**
- **Behavioural trends over time**

These features are used as inputs to the recommendation models shown later in the app.
""")

# ---------- What the data is NOT ----------
st.header("What the data does NOT represent")

st.markdown("""
It is important to clarify that the data shown:
- Does **not** represent real customers
- Does **not** reflect actual business performance
- Is **not** intended for operational or commercial use

The focus of this study is on **explainability and trust**, not predictive accuracy.
""")

# ---------- Ethics & transparency ----------
st.header("Ethical considerations")

st.markdown("""
Using an openly available Kaggle dataset, combined with anonymisation and adaptation,
ensures that this study:
- Complies with ethical research standards
- Poses no privacy risk to participants
- Maintains transparency in data sourcing

All data usage aligns with the information provided in the
**Participant Information Sheet**.
""")

st.page_link(
    "pages/1_Participant_Information.py",
    label="📄 View Participant Information Sheet",
    use_container_width=False
)

st.markdown("---")

st.info(
    "When you are ready, please continue to the next page to learn about the models and overall results used in this study."
)
