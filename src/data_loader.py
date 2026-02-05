import pandas as pd
import os


def load_raw_data(file_path):

    df = pd.read_csv(file_path)
    # df = df.isnull().sum()
    df = df.drop(columns=["id","Unnamed: 32"])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    return df


def split_features_target(df, target_column="diagnosis"):
    """
    Splits dataset into features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def create_baseline_and_production(df, baseline_ratio=0.7):
    """
    Splits data into baseline (historical) and production (new) sets.
    """
    split_index = int(len(df) * baseline_ratio)

    baseline_df = df.iloc[:split_index]
    production_df = df.iloc[split_index:]

    return baseline_df, production_df


if __name__ == "__main__":
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build absolute path to dataset
    data_path = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")

    data = load_raw_data(data_path)
    # print(data.head())
    # print(data["diagnosis"].value_counts())

    baseline_df, production_df = create_baseline_and_production(data)

    X_base, y_base = split_features_target(baseline_df)
    X_prod, y_prod = split_features_target(production_df)

    print("Baseline shape:", X_base.shape, y_base.shape)
    print("Production shape:", X_prod.shape, y_prod.shape)