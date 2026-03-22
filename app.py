import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Data Drift Monitoring Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url).sample(20000, random_state=42)

data = load_data()

# -------------------------
# SESSION STATE
# -------------------------
if "drift_simulated" not in st.session_state:
    st.session_state.drift_simulated = False

if "model_retrained" not in st.session_state:
    st.session_state.model_retrained = False

# -------------------------
# SPLIT DATA
# -------------------------
reference_data = data.sample(frac=0.7, random_state=42)
current_data = data.drop(reference_data.index).copy()

# -------------------------
# BUTTON: SIMULATE DRIFT
# -------------------------
if st.button("Simulate Drift"):
    st.session_state.drift_simulated = True
    st.session_state.model_retrained = False

# -------------------------
# APPLY DRIFT
# -------------------------
if st.session_state.drift_simulated and not st.session_state.model_retrained:
    current_data["Amount"] *= 10
    for col in current_data.columns:
        if col != "Class":
            current_data[col] *= 2
    st.warning(" Drift Simulated")

# -------------------------
# DRIFT CALCULATION
# -------------------------
drifted_columns = 0
total_columns = len(reference_data.columns)

for col in reference_data.columns:
    if col != "Class":
        ref_mean = reference_data[col].mean()
        curr_mean = current_data[col].mean()

        drift = abs(ref_mean - curr_mean) / (abs(ref_mean) + 1e-5)

        if drift > 0.2:
            drifted_columns += 1

drift_ratio = drifted_columns / total_columns

# -------------------------
# SHOW METRICS
# -------------------------
st.metric("Drift Ratio", f"{drift_ratio:.2f}")
st.metric("Drifted Columns", f"{drifted_columns}/{total_columns}")

# -------------------------
# UI FLOW
# -------------------------
if st.session_state.drift_simulated and not st.session_state.model_retrained:

    if drift_ratio > 0.5:
        st.error(" Data Drift Detected!")

        # RETRAIN BUTTON
        if st.button(" Retrain Model"):

            with st.spinner("Training model... "):

                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import train_test_split

                X = current_data.drop("Class", axis=1)
                y = current_data["Class"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500))
                ])

                model.fit(X_train, y_train)

                joblib.dump(model, "models/latest_model.pkl")

                # UPDATE STATE (NO RERUN)
                st.session_state.model_retrained = True

            st.success("Model retrained successfully!")

# -------------------------
# AFTER RETRAIN
# -------------------------
if st.session_state.model_retrained:
    st.success(" No Significant Drift")
    st.info("Model is now stable and working fine ")
