import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Placement Dashboard", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
log_model = pickle.load(open("logistic_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# LOAD DATA
# -----------------------------
conn = sqlite3.connect("university.db")
df = pd.read_sql("SELECT * FROM students", conn)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 Placement Prediction Dashboard")
st.markdown("### Analyze your chances & improve smartly")

# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("📥 Enter Details")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)
internships = st.sidebar.slider("Internships", 0, 10, 1)
projects = st.sidebar.slider("Projects", 0, 10, 2)

aptitude = st.sidebar.slider("Aptitude", 40, 100, 70)
coding = st.sidebar.slider("Coding", 30, 100, 70)
communication = st.sidebar.slider("Communication", 40, 100, 70)

# -----------------------------
# BUTTON
# -----------------------------
if st.sidebar.button("🚀 Predict"):

    # -----------------------------
    # PREPARE DATA
    # -----------------------------
    data = pd.DataFrame({
        "cgpa":[cgpa],
        "backlogs":[backlogs],
        "internships":[internships],
        "projects":[projects],
        "aptitude_score":[aptitude],
        "coding_score":[coding],
        "communication":[communication]
    })

    data_scaled = scaler.transform(data)
    prob = log_model.predict_proba(data_scaled)[0][1]
    percentage = prob * 100

    # -----------------------------
    # PREDICTION OUTPUT
    # -----------------------------
    st.subheader("🎯 Placement Probability")
    st.success(f"{percentage:.2f}% chance of placement")

    st.progress(int(percentage))

    # -----------------------------
    # STRENGTH & WEAKNESS
    # -----------------------------
    st.subheader("📊 Strengths & Weaknesses")

    strengths = []
    weaknesses = []

    if cgpa >= df['cgpa'].mean():
        strengths.append("CGPA")
    else:
        weaknesses.append("CGPA")

    if coding >= df['coding_score'].mean():
        strengths.append("Coding")
    else:
        weaknesses.append("Coding")

    if aptitude >= df['aptitude_score'].mean():
        strengths.append("Aptitude")
    else:
        weaknesses.append("Aptitude")

    if internships >= df['internships'].mean():
        strengths.append("Internships")
    else:
        weaknesses.append("Internships")
