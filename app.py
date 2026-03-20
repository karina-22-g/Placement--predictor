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



import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Feature Graphs", layout="wide")

st.title("📊 Feature vs Placement Analysis")

# -----------------------------
# LOAD DATA
# -----------------------------
conn = sqlite3.connect("university.db")
df = pd.read_sql("SELECT * FROM students", conn)

# -----------------------------
# FUNCTION → LINE GRAPH
# -----------------------------
def plot_line(feature):
    grouped = df.groupby(feature)["placed"].mean().reset_index()
    grouped["placement_rate"] = grouped["placed"] * 100

    fig = px.line(
        grouped,
        x=feature,
        y="placement_rate",
        markers=True,
        title=f"{feature.upper()} vs Placement Rate (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FUNCTION → HISTOGRAM
# -----------------------------
def plot_hist(feature):
    fig = px.histogram(
        df,
        x=feature,
        color="placed",
        barmode="overlay",
        title=f"{feature.upper()} Distribution vs Placement"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# HISTOGRAMS
# -----------------------------
st.subheader("CGPA vs Placement")
plot_hist("cgpa")

st.subheader("Coding Score vs Placement")
plot_hist("coding_score")

st.subheader("Aptitude Score vs Placement")
plot_hist("aptitude_score")

st.subheader("Communication vs Placement")
plot_hist("communication")

# -----------------------------
# LINE GRAPHS
# -----------------------------
st.subheader("Backlogs vs Placement")
plot_line("backlogs")

st.subheader("Internships vs Placement")
plot_line("internships")

st.subheader("Projects vs Placement")
plot_line("projects")
