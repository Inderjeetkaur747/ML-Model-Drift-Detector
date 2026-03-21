import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
# SESSION STATE
# -------------------------
if "model_updated" not in st.session_state:
    st.session_state.model_updated = False

if "simulate_drift" not in st.session_state:
    st.session_state.simulate_drift = False

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")

run_drift = st.sidebar.button("🔍 Run Drift Detection")
st.session_state.simulate_drift = st.sidebar.checkbox(
    "⚠️ Simulate Drift", value=st.session_state.simulate_drift
)

drift_threshold = st.sidebar.slider("📉 Drift Threshold", 0.01, 1.0, 0.2, 0.01)

# -------------------------
# DATA SPLIT
# -------------------------
reference_data = data.iloc[:30000]
current_data = data.iloc[30000:].copy()

# -------------------------
# SIMULATE DRIFT
# -------------------------
if st.session_state.simulate_drift:
    np.random.seed(42)
    current_data["Amount"] *= 3
    current_data["Time"] += 50000
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

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Features", len(drift_df))
    col2.metric("📉 Avg Drift Score", f"{drift_ratio:.3f}")
    col3.metric("🚨 Drifted Features", (drift_df["Drift Score"] > drift_threshold).sum())

    # -------------------------
    # DECISION LOGIC
    # -------------------------
    st.subheader("🧠 Model Decision")

    if drift_ratio > drift_threshold and not st.session_state.model_updated:
        st.error("🚨 Drift Detected!")
        st.warning("⚠️ Model retraining required")

        # 🔥 SHOW BUTTON ONLY WHEN NEEDED
        if st.button("🔄 Retrain Model"):
            st.info("Training model...")

            X = data.drop("Class", axis=1)
            y = data["Class"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)

            joblib.dump(model, "models/latest_model.pkl")
            st.session_state.model_updated = True
            st.session_state.simulate_drift = False  # reset drift simulation

            st.success("✅ Model retrained successfully!")
            st.experimental_rerun()  # rerun to refresh dashboard with latest state

    else:
        st.success("✅ No Significant Drift")
        st.info("Model is stable and working fine 🚀")

    # -------------------------
    # VISUALS
    # -------------------------
    st.subheader("🔥 Top Drifted Features")
    st.dataframe(drift_df.head(10))

    st.subheader("📈 Feature-wise Drift")
    st.bar_chart(drift_df.set_index("Feature"))

    st.subheader("📊 Distribution Comparison")
    feature = st.selectbox("Select Feature", drift_df["Feature"])
    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()
    st.pyplot(fig)
