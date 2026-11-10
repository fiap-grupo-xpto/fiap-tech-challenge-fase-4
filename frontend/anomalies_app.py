"""Optional standalone UI for item 3."""
import os
import streamlit as st
from anomalies_panel import render_anomalies

st.set_page_config(page_title="Fase 4 — Anomalias", page_icon="🔔", layout="wide")
render_anomalies(os.getenv("API_URL", "http://localhost:8888"))
