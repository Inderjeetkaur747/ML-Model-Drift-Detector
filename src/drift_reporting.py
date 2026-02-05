def generate_feature_drift_report(feature_drift_df):
    """
    Cleans and summarizes feature drift results.
    """
    report = feature_drift_df.copy()

    report["ks_statistic"] = report["ks_statistic"].astype(float)
    report["p_value"] = report["p_value"].astype(float)
    report["drift_detected"] = report["drift_detected"].astype(bool)

    total_features = len(report)
    drifted_features = report["drift_detected"].sum()

    summary = {
        "total_features": int(total_features),
        "drifted_features": int(drifted_features),
        "drift_percentage": float(
            round((drifted_features / total_features) * 100, 2)
        )
    }

    return report, summary


def check_drift_alert(drift_summary, feature_drift_threshold=10):

    """
    Triggers alert if too many features drift.
    """
    if drift_summary["drift_percentage"] >= feature_drift_threshold:
        return True
    return False


if __name__ == "__main__":
    import os
    from data_loader import load_raw_data, create_baseline_and_production, split_features_target
    from drift_detection import detect_feature_drift

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")

    data = load_raw_data(data_path)
    baseline_df, production_df = create_baseline_and_production(data)

    X_base, _ = split_features_target(baseline_df)
    X_prod, _ = split_features_target(production_df)

    feature_drift_df = detect_feature_drift(X_base, X_prod)

    report, summary = generate_feature_drift_report(feature_drift_df)

    alert = check_drift_alert(summary)

    print("\nDRIFT SUMMARY:")
    print(summary)

    print("\nALERT TRIGGERED:", alert)
