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
retrain_model = st.sidebar.button("🔄 Retrain Model")
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
# SIMULATE DRIFT (FIXED)
# -------------------------
if simulate_drift:
    np.random.seed(42)
    drift_cols = ["Amount", "Time"]

    for col in drift_cols:
        if col in current_data.columns:
            current_data[col] = current_data[col] * np.random.uniform(1.5, 3.0)

    st.warning("⚠️ Drift Simulation Enabled (Amount & Time shifted)")

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
    # ALERT
    # -------------------------
    if drift_ratio > drift_threshold:
        st.error("🚨 Data Drift Detected!")
    else:
        st.success("✅ No Significant Drift")

    # -------------------------
    # TOP DRIFTED FEATURES
    # -------------------------
    st.subheader("🔥 Top Drifted Features")

    top_drift = drift_df.head(10)
    st.dataframe(top_drift)

    # -------------------------
    # BAR CHART
    # -------------------------
    st.subheader("📈 Feature-wise Drift")

    st.bar_chart(drift_df.set_index("Feature"))

    # -------------------------
    # BEFORE vs AFTER GRAPH
    # -------------------------
    st.subheader("📊 Before vs After Distribution")

    feature = st.selectbox("Select Feature", drift_df["Feature"])

    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()

    st.pyplot(fig)

# -------------------------
# RETRAIN MODEL
# -------------------------
if retrain_model:

    st.subheader("🔄 Model Retraining")

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

    st.success("✅ Model retrained and saved!")

# -------------------------
# MODEL STATUS
# -------------------------
st.sidebar.subheader("🤖 Model Status")

try:
    joblib.load("models/latest_model.pkl")
    st.sidebar.success("Model Loaded ✅")
except:
    st.sidebar.warning("Model not found")
