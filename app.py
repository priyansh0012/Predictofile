import streamlit as st
import pickle
import numpy as np

# Load model
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

age = st.slider("Age", 15, 30)
attendance = st.slider("Attendance %", 0, 100)
marks = st.slider("Marks", 0, 100)
study_hours = st.slider("Study Hours", 0.0, 10.0)
income = st.slider("Parent Income", 10000, 100000)

if st.button("Predict"):
    data = np.array([[age, attendance, marks, study_hours, income]])
    result = model.predict(data)[0]

    if result == 1:
        st.error("⚠️ High Risk of Dropout")
    else:
        st.success("✅ Likely to Continue")