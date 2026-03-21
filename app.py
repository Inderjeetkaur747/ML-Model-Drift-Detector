import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("📊 Data Drift Monitoring Dashboard")

# ---------------------------
# LOAD DATA (from URL)
# ---------------------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)
    return data.sample(50000)  # reduce size for deployment

data = load_data()

# Split into reference & current
reference_data = data.iloc[:30000]
current_data = data.iloc[30000:]

st.write("Reference Data Shape:", reference_data.shape)
st.write("Current Data Shape:", current_data.shape)

# ---------------------------
# SIMPLE DRIFT LOGIC
# ---------------------------
def calculate_drift(reference, current):
    drift_scores = []

    for col in reference.columns:
        if reference[col].dtype != "object":
            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()

            drift = abs(ref_mean - curr_mean)
            drift_scores.append(drift)

    drift_ratio = np.mean(drift_scores)
    return drift_ratio

# ---------------------------
# CALCULATE DRIFT
# ---------------------------
drift_ratio = calculate_drift(reference_data, current_data)

st.subheader("📈 Drift Results")
st.write(f"Drift Ratio: {drift_ratio:.4f}")

if drift_ratio > 0.5:
    st.error("🚨 ALERT: Data Drift Detected! Retrain model!")
else:
    st.success("✅ No significant drift.")

# ---------------------------
# LOAD MODEL (OPTIONAL)
# ---------------------------
try:
    model = joblib.load("models/latest_model.pkl")
    st.success("Model loaded successfully")
except:
    st.warning("Model not found (optional for demo)")
