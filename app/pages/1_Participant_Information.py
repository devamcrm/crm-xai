import streamlit as st
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(
    page_title="Participant Information Sheet",
    layout="wide"
)

# ---------- Logo ----------
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "uol_logo.svg"

st.image(str(LOGO_PATH), width=160)
st.markdown("---")

# ---------- Page Title ----------
st.title("Participant Information Sheet")

st.markdown("""
**Title of Study:** Building Business Trust in CRM Recommendations with SHAP-powered Explainable AI  
**Version:** 1.0  
**Date:** 09/09/2025
""")

# ---------- Content ----------
st.header("Invitation to take part")
st.markdown("""
You are being invited to participate in a research study. Before you decide, it is important to understand why the research is being done and what it will involve. Please read the information carefully and feel free to ask questions. You are welcome to discuss it with friends, family, or colleagues before making your decision. Participation is voluntary, and you should only take part if you want to.
""")

st.header("What is the purpose of the study?")
st.markdown("""
This study aims to explore whether providing clear explanations for AI-generated recommendations in Customer Relationship Management (CRM) tools can increase trust and willingness to use them. Many AI tools currently act as “black boxes” with little transparency. We will test a version of a CRM recommendation tool that includes explanations (using SHAP – SHapley Additive exPlanations) and compare it to one without explanations.
""")

st.header("Why have I been chosen?")
st.markdown("""
You have been invited because you are a CRM, marketing, or Martech professional with experience using AI-based recommendation features. Around 10–15 participants will take part in the interview stage.
""")

st.header("Do I have to take part?")
st.markdown("""
No. Participation is voluntary, and you can withdraw at any time without giving a reason. Withdrawing will not affect you in any way.
""")

st.header("What will happen if I take part?")
st.markdown("""
If you agree, you will be asked to:

- Take part in a video call interview (about 30 minutes) using Zoom or Microsoft Teams.
- Review two versions of a prototype recommendation tool — one with explanations and one without.
- Share your thoughts on trust, transparency, and usability.

The interview will be audio-recorded (with your permission) to help with accurate notetaking.
""")

st.header("How will my data be used?")
st.markdown("""
The University processes personal data as part of its research and teaching activities in accordance with the lawful basis of ‘public task’, and in accordance with the University’s purpose of “advancing education, learning and research for the public benefit. 

Under UK data protection legislation, the University acts as the Data Controller for personal data collected as part of the University’s research. The Dissertation Advisor, Dr Haitham Hussien, acts as the Data Processor for this study, and any queries relating to the handling of your personal data can be sent to H.Hussien@liverpool.ac.uk.
""")

st.markdown("""
**Further information on how your data will be used can be found below:**

- **Data Collection:** Your name and email will be collected only if you volunteer for the interview.
- **Data Storage:** Data will be stored securely on encrypted University-approved platforms.
- **Data Retention:** Recordings will be deleted after transcription. Anonymised data will be kept for up to five years.
- **Data Protection:** Data will be anonymised before analysis so you cannot be identified.
- **Data Access:** Only the researcher and supervisor will have access.
- **Data Transfer:** No personal data will be transferred outside secure systems.
- **Data Archival:** No data would be archived. Your data will only be used for this MS project and will be securely destroyed after the study.
""")

st.header("Expenses and payments")
st.markdown("There are no payments or reimbursements for participation.")

st.header("Are there any risks in taking part?")
st.markdown("There are no anticipated risks. You may skip any question you prefer not to answer.")

st.header("Are there any benefits?")
st.markdown("While there is no direct personal benefit, your insights will contribute to making AI tools more transparent and trustworthy for business users.")

st.header("What will happen to the results?")
st.markdown("Results will be used in an MSc dissertation, academic presentations, and potentially published research. You will not be identifiable in any report.")

st.header("What if I want to withdraw?")
st.markdown("You can withdraw at any point before your data is anonymised by contacting the researcher. Once anonymised, it will no longer be possible to remove your data.")

st.header("What if I am unhappy or have a problem?")
st.markdown("""
If you are unhappy, or if there is a problem, please feel free to let us know by contacting Devam Saxena (WA: +60 12 843 3826) and we will try to help. If you remain unhappy or have a complaint which you feel you cannot come to us with then you should contact the Research Ethics and Integrity Office at ethics@liv.ac.uk. When contacting the Research Ethics and Integrity Office, please provide details of the name or description of the study (so that it can be identified), the researcher(s) involved, and the details of the complaint you wish to make.

The University strives to maintain the highest standards of rigour in the processing of your data. However, if you have any concerns about the way in which the University processes your personal data, it is important that you are aware of your right to lodge a complaint with the Information Commissioner's Office by calling 0303 123 1113.
""")

st.header("Who can I contact for more information?")
st.markdown("""
**Researcher:** Devam Saxena – MS in Data Science and Artificial Intelligence, University of Liverpool – D.Saxena2@liverpool.ac.uk  
**Supervisor:** Dr Haitham Hussien – University of Liverpool – H.Hussien@liverpool.ac.uk
""")
