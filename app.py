import sys
!{sys.executable} -m pip install streamlit
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Placement Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("placement_model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
conn = sqlite3.connect("university.db")
df = pd.read_sql("SELECT * FROM students", conn)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 AI Placement Prediction Dashboard")
st.markdown("### Analyze, Predict & Improve Student Placement Chances")
st.markdown("---")

# -----------------------------
# KPI CARDS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📊 Total Students", len(df))
col2.metric("🎯 Placement Rate", f"{(df['pass'].mean()*100):.2f}%")
col3.metric("📚 Avg Study Hours", f"{df['study_hours'].mean():.2f}")

st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Data Insights", "Model Insights"]
)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if menu == "Prediction":

    st.subheader("🎯 Enter Student Details")

    attendance = st.slider("Attendance (%)", 0, 100)
    study_hours = st.slider("Study Hours per Day", 0, 10)
    assignments = st.slider("Assignments Completed", 0, 10)
    internal_marks = st.slider("Internal Marks", 0, 100)

    if st.button("Predict Placement"):

        data = np.array([[attendance, study_hours, assignments, internal_marks]])

        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]

        st.subheader("📊 Prediction Result")

        st.progress(int(probability * 100))
        st.write(f"### Placement Probability: {probability*100:.2f}%")

        if prediction[0] == 1:
            st.success("✅ Likely to be Placed")
        else:
            st.error("❌ Less Chance of Placement")

        # -----------------------------
        # IMPROVEMENT SUGGESTIONS
        # -----------------------------
        st.subheader("📌 Improvement Suggestions")

        if attendance < 75:
            st.warning("📉 Improve Attendance")

        if study_hours < 3:
            st.warning("📚 Increase Study Hours")

        if assignments < 3:
            st.warning("📝 Complete More Assignments")

        if internal_marks < 60:
            st.warning("📊 Improve Internal Marks")

        if attendance >= 75 and study_hours >= 3 and assignments >= 3 and internal_marks >= 60:
            st.success("🔥 Strong Profile - Keep it up!")

# -----------------------------
# DATA INSIGHTS SECTION
# -----------------------------
elif menu == "Data Insights":

    st.subheader("📊 Data Insights Dashboard")

    col1, col2 = st.columns(2)

    # Placement Distribution
    with col1:
        fig1 = px.pie(
            df,
            names='pass',
            hole=0.5,
            title="Placement Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Study Hours vs Placement
    with col2:
        fig2 = px.box(
            df,
            x="pass",
            y="study_hours",
            color="pass",
            title="Study Hours vs Placement"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Attendance Distribution
    fig3 = px.histogram(
        df,
        x="attendance",
        color="pass",
        title="Attendance Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# MODEL INSIGHTS SECTION
# -----------------------------
elif menu == "Model Insights":

    st.subheader("🤖 Model Insights")

    st.write("Model Used: Logistic Regression")

    st.write("### 📊 Recall-Focused Model")
    st.write("""
    Recall is prioritized because missing a potentially placed student
    (false negative) is more harmful than predicting placement incorrectly.
    """)

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("📈 Feature Importance")

    features = ["attendance","study_hours","assignments","internal_marks"]

    try:
        importance = model.feature_importances_
    except:
        importance = model.coef_[0]

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    fig4 = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        color="Importance",
        title="Feature Importance"
    )

    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("🚀 Developed for Hackathon 3 | ML Deployment & Dashboard")
