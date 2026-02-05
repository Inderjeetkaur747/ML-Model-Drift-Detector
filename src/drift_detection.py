import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_prediction_drift(
    model,
    scaler,
    X_baseline,
    X_production,
    threshold=0.05
):
    """
    Detects prediction drift using KS test.
    """
    # Scale data using baseline scaler
    X_base_scaled = scaler.transform(X_baseline)
    X_prod_scaled = scaler.transform(X_production)

    # Get model predictions (probabilities)
    base_preds = model.predict_proba(X_base_scaled)[:, 1]
    prod_preds = model.predict_proba(X_prod_scaled)[:, 1]

    # Perform KS test
    ks_stat, p_value = ks_2samp(base_preds, prod_preds)

    drift_detected = p_value < threshold

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "drift_detected": drift_detected
    }


def detect_feature_drift(X_baseline,
    X_production,
    threshold=0.05
):
    """
    Detects feature drift using KS test for each feature.
    """
    drift_results = []

    for feature in X_baseline.columns:
        base_values = X_baseline[feature]
        prod_values = X_production[feature]

        ks_stat, p_value = ks_2samp(base_values, prod_values)

        drift_detected = p_value < threshold

        drift_results.append({
            "feature": feature,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": drift_detected
        })

    return pd.DataFrame(drift_results)



if __name__ == "__main__":
    import os
    from data_loader import load_raw_data, create_baseline_and_production, split_features_target
    from model import train_baseline_model

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")

    data = load_raw_data(data_path)
    baseline_df, production_df = create_baseline_and_production(data)

    X_base, y_base = split_features_target(baseline_df)
    X_prod, y_prod = split_features_target(production_df)

    model, scaler = train_baseline_model(X_base, y_base)

    # drift_result = detect_prediction_drift(
    #     model,
    #     scaler,
    #     X_base,
    #     X_prod
    # )

    # print("Prediction Drift Result:")
    # print(drift_result)

    # Feature drift
    feature_drift_df = detect_feature_drift(X_base, X_prod)

    print("\nFeature Drift Summary:")
    print(feature_drift_df.head())
    print("\nTotal drifted features:",
          feature_drift_df["drift_detected"].sum())