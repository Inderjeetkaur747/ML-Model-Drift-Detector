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
# SESSION STATE
# -------------------------
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")

run_drift = st.sidebar.button("🔍 Run Drift Detection")
simulate_drift = st.sidebar.checkbox("⚠️ Simulate Drift")

drift_threshold = st.sidebar.slider(
    "📉 Drift Threshold", 0.01, 1.0, 0.2, 0.01
)

# -------------------------
# DATA SPLIT
# -------------------------
reference_data = data.iloc[:30000]
current_data = data.iloc[30000:].copy()

# -------------------------
# SIMULATE DRIFT
# -------------------------
if simulate_drift and not st.session_state.model_trained:
    current_data["Amount"] *= 3
    current_data["Time"] += 50000
    st.warning("⚠️ Drift Simulation Enabled")

# -------------------------
# DRIFT FUNCTION
# -------------------------
def calculate_drift(reference, current):
    drift_scores = []

    for col in reference.columns:
        if reference[col].dtype != "object":
            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()

            drift = abs(ref_mean - curr_mean) / (abs(ref_mean) + 1e-5)
            drift_scores.append(drift)

    return np.mean(drift_scores)

# -------------------------
# MAIN LOGIC
# -------------------------
if run_drift:

    st.subheader("📊 Drift Analysis")

    drift_ratio = calculate_drift(reference_data, current_data)

    col1, col2 = st.columns(2)

    col1.metric("📊 Features", reference_data.shape[1])
    col2.metric("📉 Drift Score", f"{drift_ratio:.3f}")

    # -------------------------
    # DECISION
    # -------------------------
    if drift_ratio > drift_threshold and not st.session_state.model_trained:

        st.error("🚨 Drift Detected")
        st.warning("⚠️ Retraining required")

        # -------------------------
        # RETRAIN BUTTON
        # -------------------------
        if st.button("🔄 Retrain Model"):

            st.info("Training model...")

            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split

            X = data.drop("Class", axis=1)
            y = data["Class"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)

            joblib.dump(model, "models/latest_model.pkl")

            # 🔥 KEY FIX
            st.session_state.model_trained = True

            st.success("✅ Model retrained successfully!")

            st.rerun()

    else:
        st.success("✅ No Significant Drift")
        st.info("Model is stable and working fine 🚀")

    # -------------------------
    # VISUAL CHECK
    # -------------------------
    st.subheader("📊 Example Feature Distribution")

    feature = "Amount"

    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()

    st.pyplot(fig)
