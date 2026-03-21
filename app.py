import streamlit as st
import pandas as pd
import json
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

st.title("Data Drift Monitoring Dashboard")

# Load data
data = pd.read_csv("data/raw/creditcard.csv")

# Split data
reference_data = data.sample(frac=0.7, random_state=42)
current_data = data.drop(reference_data.index)

# BUTTON: Simulate Drift
if st.button(" Simulate Drift"):
    current_data["Amount"] = current_data["Amount"] * 10
    for col in current_data.columns:
        if col != "Class":
            current_data[col] = current_data[col] * 2
    st.success("Drift simulated!")

# Run drift detection
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save JSON
report_dict = report.as_dict()

# Extract drift summary
result = report_dict['metrics'][0]['result']
drifted_columns = result['number_of_drifted_columns']
total_columns = result['number_of_columns']
drift_ratio = drifted_columns / total_columns

# Show metrics
st.metric("Drift Ratio", f"{drift_ratio:.2f}")
st.metric("Drifted Columns", f"{drifted_columns}/{total_columns}")

# Alert
if drift_ratio > 0.5:
    st.error(" Data Drift Detected!")

    # Retrain button
    if st.button("Retrain Model"):
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
            ("clf", LogisticRegression(max_iter=2000))
        ])

        model.fit(X_train, y_train)

        joblib.dump(model, "../models/latest_model.pkl")

        st.success(" Model retrained and saved!")

else:
    st.success("No significant drift")