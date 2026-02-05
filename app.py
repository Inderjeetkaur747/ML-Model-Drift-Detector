import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_data, split_features_target
from src.drift_detection import detect_feature_drift
from src.drift_reporting import generate_feature_drift_report, check_drift_alert

st.set_page_config(page_title="ML Drift Detector", layout="wide")

st.title("ML Model Drift Detection Dashboard")

st.write("""
This dashboard detects **feature drift** between baseline and production data
using statistical tests.
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
baseline_path = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")

st.subheader("Baseline Dataset")
baseline_df = load_raw_data(baseline_path)
st.success("Baseline data loaded successfully")
st.dataframe(baseline_df.head())

st.subheader("Upload Production Data")

uploaded_file = st.file_uploader(
    "Upload production CSV",
    type=["csv"]
)

if uploaded_file:
    prod_df = pd.read_csv(uploaded_file)
    st.success("Production data uploaded")
    st.dataframe(prod_df.head())


if uploaded_file and st.button("Run Drift Detection"):
    X_base, _ = split_features_target(baseline_df)
    X_prod, _ = split_features_target(prod_df)

    feature_drift_df = detect_feature_drift(X_base, X_prod)
    report, summary = generate_feature_drift_report(feature_drift_df)

    alert = check_drift_alert(summary)

    st.subheader("Drift Summary")
    st.json(summary)

    if alert:
        st.error("🚨 ALERT: Significant feature drift detected!")
    else:
        st.success("✅ No significant drift detected")

    st.subheader("🔍 Feature Drift Details")
    st.dataframe(report)
