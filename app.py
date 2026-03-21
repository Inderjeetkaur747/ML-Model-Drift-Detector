import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.title(" ML Model Monitoring Dashboard")

# LOAD DATA

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)
    return data.sample(50000)

data = load_data()

reference_data = data.iloc[:30000]
current_data = data.iloc[30000:]


# DRIFT FUNCTION

def calculate_drift(reference, current):
    drift_dict = {}

    for col in reference.columns:
        if reference[col].dtype != "object":
            ref_mean = reference[col].mean()
            curr_mean = current[col].mean()
            drift = abs(ref_mean - curr_mean)
            drift_dict[col] = drift

    return drift_dict


# SIDEBAR CONTROLS

st.sidebar.header("Controls")

run_drift = st.sidebar.button(" Run Drift Detection")
retrain_model = st.sidebar.button(" Retrain Model")


# MAIN DASHBOARD

if run_drift:

    st.subheader(" Drift Analysis")

    drift_dict = calculate_drift(reference_data, current_data)

    drift_values = np.array(list(drift_dict.values()))
    drift_ratio = np.mean(drift_values) / (np.std(drift_values) + 1e-5)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Features", len(drift_dict))
    col2.metric("Drift Score", f"{drift_ratio:.2f}")
    col3.metric("Drifted Features", sum(v > 0.5 for v in drift_dict.values()))

    if drift_ratio > 0.5:
        st.error(" Data Drift Detected!")
    else:
        st.success(" No Significant Drift")

    # Bar chart
    st.subheader(" Feature Drift")

    drift_df = pd.DataFrame({
        "Feature": list(drift_dict.keys()),
        "Drift": list(drift_dict.values())
    }).sort_values(by="Drift", ascending=False)

    st.bar_chart(drift_df.set_index("Feature"))

    # Distribution comparison
    st.subheader("Feature Distribution")

    feature = st.selectbox("Select Feature", drift_df["Feature"])

    fig, ax = plt.subplots()
    ax.hist(reference_data[feature], bins=50, alpha=0.5, label="Reference")
    ax.hist(current_data[feature], bins=50, alpha=0.5, label="Current")
    ax.legend()

    st.pyplot(fig)


# RETRAIN BUTTON

if retrain_model:
    st.subheader(" Model Retraining")

    st.info("Training model...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/latest_model.pkl")

    st.success(" Model retrained and saved!")


# MODEL STATUS

st.sidebar.subheader(" Model Status")

try:
    joblib.load("models/latest_model.pkl")
    st.sidebar.success("Model Loaded ")
except:
    st.sidebar.warning("Model not found")

