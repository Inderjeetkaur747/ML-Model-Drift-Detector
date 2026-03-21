import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.title("🚀 ML Model Monitoring Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)
    return data.sample(50000, random_state=42)

data = load_data()

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Controls")

run_drift = st.sidebar.button("🔍 Run Drift Detection")
simulate_drift = st.sidebar.checkbox("⚠️ Simulate Drift")

drift_threshold = st.sidebar.slider(
    "📉 Drift Threshold",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.01
)

# -------------------------
# SPLIT DATA
# -------------------------
reference_data = data.iloc[:30000]
current_data = data.iloc[30000:].copy()

# -------------------------
# SIMULATE DRIFT
# -------------------------
if simulate_drift:
    np.random.seed(42)
    drift_cols = ["Amount", "Time"]

    for col in drift_cols:
        current_data[col] = current_data[col] * np.random.uniform(1.5, 3.0)

    st.warning("⚠️ Drift Simulation Enabled")

# -------------------------
# DRIFT FUNCTION
# -------------------------
def calculate_drift(reference, current):
    drift_dict = {}

    for col in reference.columns:
        if reference[col].dtype != "object":
            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()
            drift = abs(ref_mean - curr_mean) / (abs(ref_mean) + 1e-5)
            drift_dict[col] = drift

    return drift_dict

# -------------------------
# MODEL STATUS PANEL
# -------------------------
st.sidebar.subheader("🤖 Model Status")

try:
    joblib.load("models/latest_model.pkl")
    st.sidebar.success("Model Loaded & Active ✅")
except:
    st.sidebar.warning("No production model found")

# -------------------------
# RUN DRIFT
# -------------------------
if run_drift:

    st.subheader("📊 Drift Analysis")

    drift_dict = calculate_drift(reference_data, current_data)

    drift_df = pd.DataFrame({
        "Feature": list(drift_dict.keys()),
        "Drift Score": list(drift_dict.values())
    }).sort_values(by="Drift Score", ascending=False)

    drift_ratio = drift_df["Drift Score"].mean()

    # -------------------------
    # METRICS
    # -------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Features", len(drift_df))
    col2.metric("📉 Avg Drift Score", f"{drift_ratio:.3f}")
    col3.metric(
        "🚨 Drifted Features",
        (drift_df["Drift Score"] > drift_threshold).sum()
    )

    # -------------------------
    # APPROVAL LOGIC
    # -------------------------
    st.subheader("🧠 Model Validation Decision")

    if drift_ratio > drift_threshold:
        st.error("🚨 Significant Drift Detected")
        st.warning("⚠️ Model retraining is recommended before deployment.")
    else:
        st.success("✅ No Significant Drift")
        st.info("Model performance is expected to remain stable. Retraining not required.")

    # -------------------------
    # TOP DRIFTED FEATURES
    # -------------------------
    st.subheader("🔥 Top Drifted Features")

    st.dataframe(drift_df.head(10))

    # -------------------------
    # DRIFT CHART
    # -------------------------
    st.subheader("📈 Feature-wise Drift")

    st.bar_chart(drift_df.set_index("Feature"))

    # -------------------------
    # DISTRIBUTION COMPARISON
    # -------------------------
    st.subheader("📊 Before vs After Distribution")

    feature = st.selectbox("Select Feature", drift_df["Feature"])

    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()

    st.pyplot(fig)
