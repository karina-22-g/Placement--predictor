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
# LINE GRAPH (3 FEATURES IN ONE)
# -----------------------------
st.subheader("📈 Backlogs, Internships, Projects vs Placement")

features_line = ["backlogs", "internships", "projects"]
combined = pd.DataFrame()

for feature in features_line:
    temp = df.groupby(feature)["placed"].mean().reset_index()
    temp["placement_rate"] = temp["placed"] * 100
    temp["feature"] = feature
    temp.rename(columns={feature: "value"}, inplace=True)
    combined = pd.concat([combined, temp])

fig1 = px.line(
    combined,
    x="value",
    y="placement_rate",
    color="feature",
    markers=True,
    title="📈 Placement Trend for Key Features"
)

fig1.update_layout(template="plotly_dark")

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# BAR GRAPH (COLORFUL + CLEAN)
# -----------------------------
st.subheader("🎯 Key Features Impact on Placement")

features_bar = ["cgpa", "coding_score", "aptitude_score", "communication"]

# calculate averages
df_grouped = df.groupby("placed")[features_bar].mean().reset_index()

# reshape
df_melted = df_grouped.melt(
    id_vars="placed",
    var_name="feature",
    value_name="average_value"
)

# rename labels for better understanding
df_melted["placed"] = df_melted["placed"].map({0: "Not Placed", 1: "Placed"})
df_melted["feature"] = df_melted["feature"].replace({
    "cgpa": "CGPA",
    "coding_score": "Coding",
    "aptitude_score": "Aptitude",
    "communication": "Communication"
})

# colorful bar chart
fig2 = px.bar(
    df_melted,
    x="feature",
    y="average_value",
    color="placed",
    barmode="group",
    text_auto=True,
    title="🎯 Average Performance: Placed vs Not Placed",
    color_discrete_map={
        "Placed": "#00FFAA",       # green
        "Not Placed": "#FF4B4B"    # red
    }
)

fig2.update_layout(
    template="plotly_dark",
    xaxis_title="Features",
    yaxis_title="Average Score",
    legend_title="Placement Status"
)

st.plotly_chart(fig2, use_container_width=True)
