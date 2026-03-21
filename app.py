import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Drift Dashboard", layout="wide")

st.title("📊 ML Model Monitoring Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)
    return data.sample(50000)

data = load_data()

reference_data = data.iloc[:30000]
current_data = data.iloc[30000:]

# -------------------------
# DRIFT CALCULATION
# -------------------------
def calculate_drift(reference, current):
    drift_dict = {}

    for col in reference.columns:
        if reference[col].dtype != "object":
            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()
            drift = abs(ref_mean - curr_mean)

            drift_dict[col] = drift

    return drift_dict

drift_dict = calculate_drift(reference_data, current_data)

# Normalize drift (for better scale)
drift_values = np.array(list(drift_dict.values()))
drift_ratio = np.mean(drift_values) / (np.std(drift_values) + 1e-5)

# -------------------------
# TOP METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📊 Total Features", len(drift_dict))
col2.metric("📉 Avg Drift Score", f"{drift_ratio:.2f}")
col3.metric("🚨 Drifted Features", sum(v > 0.5 for v in drift_dict.values()))

# -------------------------
# ALERT
# -------------------------
if drift_ratio > 0.5:
    st.error("🚨 Data Drift Detected! Retraining Recommended")
else:
    st.success("✅ No Significant Drift")

# -------------------------
# DRIFT BAR CHART
# -------------------------
st.subheader("📈 Feature-wise Drift")

drift_df = pd.DataFrame({
    "Feature": list(drift_dict.keys()),
    "Drift Score": list(drift_dict.values())
}).sort_values(by="Drift Score", ascending=False)

st.bar_chart(drift_df.set_index("Feature"))

# -------------------------
# DISTRIBUTION COMPARISON
# -------------------------
st.subheader("📊 Feature Distribution Comparison")

feature = st.selectbox("Select Feature", drift_df["Feature"].values)

fig, ax = plt.subplots()
ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
ax.legend()

st.pyplot(fig)

# -------------------------
# MODEL STATUS
# -------------------------
st.subheader("🤖 Model Status")

try:
    model = joblib.load("models/latest_model.pkl")
    st.success("Model loaded successfully ✅")
except:
    st.warning("Model not found")
