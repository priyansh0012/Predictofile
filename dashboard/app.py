import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# ✅ FIRST Streamlit command
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

# ✅ Styling AFTER config
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #161a28;
}
</style>
""", unsafe_allow_html=True)



# Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Title
st.title("🎓 AI-Based Student Dropout Prediction System")
st.markdown("### Predict whether a student is at risk of dropping out using Machine Learning")

# Sidebar Inputs
st.sidebar.header("📊 Enter Student Details")

age = st.sidebar.slider("Age", 15, 30)
attendance = st.sidebar.slider("Attendance (%)", 0, 100)
marks = st.sidebar.slider("Marks", 0, 100)
study_hours = st.sidebar.slider("Study Hours", 0.0, 10.0)
income = st.sidebar.slider("Parent Income (₹)", 10000, 100000)

# Main Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Input Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**Attendance:** {attendance}%")
    st.write(f"**Marks:** {marks}")
    st.write(f"**Study Hours:** {study_hours}")
    st.write(f"**Parent Income:** ₹{income}")

with col2:
    st.subheader("🤖 Prediction Result")

    if st.button("Predict"):
        data = np.array([[age, attendance, marks, study_hours, income]])
        
        result = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if result == 1:
            st.error("⚠️ High Risk of Dropout")
            st.metric("Dropout Probability", f"{prob*100:.2f}%")
            st.markdown("### 🔴 Intervention Needed")
            st.write("Suggested Actions:")
            st.write("- Improve attendance")
            st.write("- Increase study hours")
            st.write("- Provide academic support")

        else:
            st.success("✅ Likely to Continue")
            st.markdown(f"**Confidence:** {(1-prob)*100:.2f}%")
            st.markdown("### 🟢 Student on Track")
            st.write("Keep maintaining performance!")

st.markdown("---")
st.subheader("📊 Feature Importance")

# Feature importance
features = ["Age", "Attendance", "Marks", "Study Hours", "Parent Income"]
importance = model.feature_importances_

# Create DataFrame
df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=True)

# Plot
fig, ax = plt.subplots()
ax.barh(df["Feature"], df["Importance"])
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance (Model Insight)")

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Priyanshu Sharma")