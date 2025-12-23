import streamlit as st
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(
    page_title="Overview & Welcome",
    layout="wide"
)

# ---------- Assets ----------
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_SVG = ASSETS_DIR / "uol_logo.svg"

# ---------- Header (Logo + Navigation) ----------
col_logo, col_spacer, col_nav = st.columns([3, 1, 2])

with col_logo:
    st.image(str(LOGO_SVG), width=180)

with col_nav:
    nav1, nav2 = st.columns(2)
    with nav1:
        st.page_link(
            "pages/1_Participant_Information.py",
            label="📄 Participant Info",
            use_container_width=True
        )
    with nav2:
        st.page_link(
            "pages/2_How_This_Works.py",
            label="➡ Continue",
            use_container_width=True
        )

st.markdown("---")

# ---------- Title ----------
st.title("Overview & Welcome")

st.markdown(
    """
Thank you for agreeing to explore this CRM recommendation engine and for sharing your views.
Your participation is greatly appreciated and contributes directly to academic research.
"""
)

# ---------- What is this app ----------
st.header("What is this app?")

st.markdown(
    """
This application accompanies an MSc research project at the **University of Liverpool**.

It demonstrates a prototype **Customer Relationship Management (CRM) recommendation engine**
and explores how explainable artificial intelligence techniques — specifically **SHAP**
(SHapley Additive exPlanations) — influence business user trust in AI-driven recommendations.
"""
)

# ---------- Why it exists ----------
st.header("Why is it created and by whom?")

st.markdown(
    """
This app has been created by **Devam Saxena** as part of the MSc in
**Data Science and Artificial Intelligence** at the **University of Liverpool**,
under the supervision of **Dr Haitham Hussien**.

The research investigates whether providing clear, understandable explanations
can improve:
- Trust in AI recommendations
- Transparency of decision-making
- Willingness to rely on AI systems in professional CRM contexts
"""
)

# ---------- Use of responses ----------
st.header("Use of your responses")

st.markdown(
    """
Anything you choose to share while using this application is used **solely for research purposes**
and is governed by the information outlined in the **Participant Information Sheet**.

You are encouraged to review this information before continuing.
"""
)

st.page_link(
    "pages/1_Participant_Information.py",
    label="📄 View Participant Information Sheet",
    use_container_width=False
)

st.markdown("---")

st.info(
    "When you are ready, click **Continue** (top-right) to learn how the experience works."
)
