import streamlit as st
from core_app import run_retention_dashboard

st.set_page_config(
    page_title="Retention Tahmin Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯  AB Retention & LTV Prediction Dashboard")

run_retention_dashboard()
