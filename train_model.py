import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = [
    "gid",
    "Kategorie",
    "label",
    "rssi_bad_ratio_recent_change",
]


def select_features(df):
    numeric = [
        # Health
        "7d_offline_duration_sec_mean",
        "14d_offline_duration_sec_mean",
        "28d_offline_duration_sec_mean",
        "7d_offline_duration_sec_max",
        "28d_offline_duration_sec_max",

        "7d_disconnection_cnt_mean",
        "14d_disconnection_cnt_mean",
        "28d_disconnection_cnt_mean",
        "7d_disconnection_cnt_active_days",
        "28d_disconnection_cnt_active_days",

        "7d_reboot_cnt_mean",
        "14d_reboot_cnt_mean",
        "28d_reboot_cnt_mean",
        "7d_reboot_cnt_active_days",
        "28d_reboot_cnt_active_days",

        "7d_reboot_duration_sec_mean",
        "28d_reboot_duration_sec_mean",

        "7d_avg_uptime_mean",
        "28d_avg_uptime_mean",

        # System
        "7d_avg_load1_mean",
        "14d_avg_load1_mean",
        "28d_avg_load1_mean",
        "7d_avg_load1_max",
        "28d_avg_load1_max",

        "7d_avg_memfree_mean",
        "28d_avg_memfree_mean",

        # Network / signal
        "7d_rssi_bad_ratio",
        "14d_rssi_bad_ratio",
        "28d_rssi_bad_ratio",

        "7d_rscp_bad_ratio",
        "14d_rscp_bad_ratio",
        "28d_rscp_bad_ratio",

        "7d_ecio_bad_ratio",
        "14d_ecio_bad_ratio",
        "28d_ecio_bad_ratio",

        # Traffic
        "7d_tx_success_mean",
        "28d_tx_success_mean",
        "7d_tx_busy_mean",
        "28d_tx_busy_mean",

        # Business outcome
        "meter_read_rate_mean",
        "meter_read_rate_min",
        "meter_read_rate_std",
        "zero_read_weeks",

        # Trends
        "offline_duration_sec_recent_change",
        "disconnection_cnt_recent_change",
        "reboot_cnt_recent_change",
        "avg_load1_recent_change",
        "avg_uptime_recent_change",
    ]

    categorical = [
        "site_type",
        "hw_model",
        "antenna_type",
        "fw_version",
        "region",
    ]

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]

    return numeric, categorical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ml_training.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    y = df["label"]

    numeric_features, categorical_features = select_features(df)

    X = df[numeric_features + categorical_features]

    print("Training dataset")
    print("----------------")
    print("Rows:", len(X))
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Total features:", X.shape[1])
    print()

    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())
    print()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=0.1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    print("5-fold cross-validation")
    print("-----------------------")
    print(
        "Accuracy:",
        round(accuracy_score(y, predictions), 3),
    )
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y, predictions), 3),
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y, probabilities), 3),
    )

    results = df[
        ["gid", "Kategorie", "label"]
    ].copy()

    results["prob_schlecht"] = probabilities

    results = results.sort_values(
        "prob_schlecht",
        ascending=False,
    )

    results.to_csv(
        "ml_cv_predictions.csv",
        index=False,
    )

    print()
    print("Top 15 cross-validation predictions:")
    print(
        results.head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()