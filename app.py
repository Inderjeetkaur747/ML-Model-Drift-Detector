import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="🚀 ML Monitoring Dashboard", layout="wide")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "status" not in st.session_state:
    st.session_state.status = "idle"  # idle, detecting_drift, retraining, done

if "model_updated" not in st.session_state:
    st.session_state.model_updated = False

if "simulate_drift" not in st.session_state:
    st.session_state.simulate_drift = False

if "drift_results" not in st.session_state:
    st.session_state.drift_results = None

# -------------------------
# TOP STATUS MESSAGE
# -------------------------
if st.session_state.status == "detecting_drift":
    st.info("🔍 Running Drift Detection...")
elif st.session_state.status == "retraining":
    st.info("⚙️ Retraining the model...")
elif st.session_state.status == "done":
    st.success("✅ Last operation completed successfully!")

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
reference_data = data.iloc[:30000]
current_data = data.iloc[30000:].copy()

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Controls")
st.session_state.simulate_drift = st.sidebar.checkbox(
    "⚠️ Simulate Drift", value=st.session_state.simulate_drift
)
drift_threshold = st.sidebar.slider("📉 Drift Threshold", 0.01, 1.0, 0.2, 0.01)
run_drift = st.sidebar.button("🔍 Run Drift Detection")

# -------------------------
# SIMULATE DRIFT
# -------------------------
if st.session_state.simulate_drift:
    np.random.seed(42)
    current_data["Amount"] *= 3
    current_data["Time"] += 50000
    st.warning("⚠️ Drift Simulation Enabled")

# -------------------------
# FUNCTION: CALCULATE DRIFT
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
# STEP 1: DATA SNAPSHOT
# -------------------------
with st.expander("📂 Data Overview"):
    st.subheader("Reference Data (First 30k rows)")
    st.dataframe(reference_data.head())
    st.subheader("Current Data (Last 20k rows)")
    st.dataframe(current_data.head())
    st.subheader("Data Stats")
    st.write(data.describe())

# -------------------------
# STEP 2 & 3: DRIFT DETECTION
# -------------------------
if run_drift or st.session_state.status in ["detecting_drift", "done"]:
    st.session_state.status = "detecting_drift"
    
    drift_dict = calculate_drift(reference_data, current_data)
    drift_df = pd.DataFrame({
        "Feature": list(drift_dict.keys()),
        "Drift Score": list(drift_dict.values())
    }).sort_values(by="Drift Score", ascending=False)
    st.session_state.drift_results = drift_df
    drift_ratio = drift_df["Drift Score"].mean()

    st.subheader("📊 Drift Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Features", len(drift_df))
    col2.metric("📉 Avg Drift Score", f"{drift_ratio:.3f}")
    col3.metric("🚨 Drifted Features", (drift_df["Drift Score"] > drift_threshold).sum())

    st.subheader("🧠 Model Status")
    if drift_ratio > drift_threshold and not st.session_state.model_updated:
        st.error("🚨 Drift Detected! Model retraining required.")
        if st.button("🔄 Retrain Model"):
            st.session_state.status = "retraining"
            st.experimental_rerun()  # trigger rerun to show "retraining" status

# -------------------------
# STEP 4: MODEL RETRAINING
# -------------------------
if st.session_state.status == "retraining":
    with st.spinner("Training model..."):
        X = data.drop("Class", axis=1)
        y = data["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        joblib.dump(model, "models/latest_model.pkl")
        st.session_state.model_updated = True
        st.session_state.simulate_drift = False
        st.session_state.status = "done"
        st.experimental_rerun()  # rerun to update dashboard

# -------------------------
# STEP 5: VISUALS
# -------------------------
if st.session_state.drift_results is not None:
    drift_df = st.session_state.drift_results
    st.subheader("🔥 Top Drifted Features")
    st.dataframe(drift_df.head(10))

    st.subheader("📈 Feature-wise Drift")
    st.bar_chart(drift_df.set_index("Feature"))

    st.subheader("📊 Feature Distribution Comparison")
    feature = st.selectbox("Select Feature to Compare", drift_df["Feature"])
    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()
    st.pyplot(fig)
