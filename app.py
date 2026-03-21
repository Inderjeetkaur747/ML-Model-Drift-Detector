import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Evidently imports
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ModuleNotFoundError:
    st.error("Evidently is not installed. Install it via `pip install evidently` in your environment.")
    st.stop()

st.title("📊 Data Drift Monitoring Dashboard")

# Load data
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)

data = load_data()

# Split data
reference_data = data.sample(frac=0.7, random_state=42)
current_data = data.drop(reference_data.index).copy()  # copy to avoid SettingWithCopyWarning

st.subheader("Preview of Current Data")
st.dataframe(current_data.head())

# --- Simulate Drift ---
if st.button("⚡ Simulate Drift"):
    current_data["Amount"] = current_data["Amount"] * 10
    for col in current_data.columns:
        if col != "Class":
            current_data[col] = current_data[col] * 2
    st.success("✅ Drift simulated!")

# --- Run Drift Detection ---
st.subheader("Data Drift Detection")
if st.button("🔍 Run Drift Detection"):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report_dict = report.as_dict()

    # Extract drift summary
    result = report_dict['metrics'][0]['result']
    drifted_columns = result['number_of_drifted_columns']
    total_columns = result['number_of_columns']
    drift_ratio = drifted_columns / total_columns

    # Show metrics
    st.metric("Drift Ratio", f"{drift_ratio:.2f}")
    st.metric("Drifted Columns", f"{drifted_columns}/{total_columns}")

    # Alert & Retrain option
    if drift_ratio > 0.5:
        st.error("⚠️ Data Drift Detected!")

        if st.button("🔄 Retrain Model"):
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
            joblib.dump(model, "latest_model.pkl")  # Use relative path for Streamlit

            st.success("✅ Model retrained and saved!")
    else:
        st.success("🎉 No significant drift detected.")

st.info("💡 Workflow: You can simulate drift → detect drift → retrain model if needed.")
