from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_baseline_model(X, y):
    """
    Trains a baseline logistic regression model with feature scaling.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled,y)
    return model, scaler


def evaluate_model(model,scaler, X, y):
    """
    Evaluates model performance.
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    accuracy = accuracy_score(y, predictions)

    return accuracy


if __name__ == "__main__":
    import os
    from data_loader import load_raw_data, create_baseline_and_production, split_features_target

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")

    data = load_raw_data(data_path)
    baseline_df, _ = create_baseline_and_production(data)

    X_base, y_base = split_features_target(baseline_df)

    model, scaler = train_baseline_model(X_base, y_base)
    acc = evaluate_model(model, scaler, X_base, y_base)

    print("Baseline model accuracy:", acc)
