import streamlit as st
import joblib
import numpy as np

model = joblib.load("placement_model.pkl")

st.title("Placement Prediction System")

st.write("Enter Student Details")

coding = st.number_input("Coding Score", 0, 100)
aptitude = st.number_input("Aptitude Score", 0, 100)
mock = st.number_input("Mock Interview Score", 0, 100)
communication = st.number_input("Communication Score", 0, 100)
projects = st.number_input("Projects Count", 0, 10)

if st.button("Predict Placement"):

    data = np.array([[coding, aptitude, mock, communication, projects]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Likely to be Placed")
    else:
        st.error("Less Chance of Placement")
