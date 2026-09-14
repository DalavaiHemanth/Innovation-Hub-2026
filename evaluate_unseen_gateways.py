import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv("ml_training.csv")

TARGET = "label"
GROUP = "gid"


# ---------------------------------------------------------
# Features used by current RF
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "7d_disconnection_cnt_mean",
    "14d_disconnection_cnt_mean",
    "28d_disconnection_cnt_mean",

    "7d_disconnection_cnt_max",
    "14d_disconnection_cnt_max",
    "28d_disconnection_cnt_max",

    "7d_disconnection_cnt_active_days",
    "14d_disconnection_cnt_active_days",
    "28d_disconnection_cnt_active_days",

    "7d_offline_duration_sec_mean",
    "14d_offline_duration_sec_mean",
    "28d_offline_duration_sec_mean",

    "7d_offline_duration_sec_max",
    "14d_offline_duration_sec_max",
    "28d_offline_duration_sec_max",

    "7d_avg_load1_mean",
    "14d_avg_load1_mean",
    "28d_avg_load1_mean",

    "7d_avg_load1_max",
    "14d_avg_load1_max",
    "28d_avg_load1_max",

    "7d_avg_uptime_mean",
    "14d_avg_uptime_mean",
    "28d_avg_uptime_mean",

    "7d_rssi_bad_ratio",
    "14d_rssi_bad_ratio",
    "28d_rssi_bad_ratio",

    "7d_rscp_bad_ratio",
    "14d_rscp_bad_ratio",
    "28d_rscp_bad_ratio",

    "7d_ecio_bad_ratio",
    "14d_ecio_bad_ratio",
    "28d_ecio_bad_ratio",

    "7d_rx_crc_bad_mean",
    "14d_rx_crc_bad_mean",
    "28d_rx_crc_bad_mean",

    "7d_tx_success_mean",
    "14d_tx_success_mean",
    "28d_tx_success_mean",

    "7d_reboot_cnt_mean",
    "14d_reboot_cnt_mean",
    "28d_reboot_cnt_mean",

    "meter_read_rate_mean",
    "meter_read_rate_std",
    "zero_read_weeks",

    "offline_duration_sec_recent_change",
    "reboot_cnt_recent_change",
]


# Current RF uses these 3 categorical features.
CURRENT_CATEGORICAL = [
    "hw_model",
    "site_type",
    "region",
]

# Additional metadata we want to test.
EXTRA_CATEGORICAL = [
    "antenna_type",
    "fw_version",
]


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def make_model(categorical_features):

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

all_features = (
    NUMERIC_FEATURES
    + CURRENT_CATEGORICAL
    + EXTRA_CATEGORICAL
)

X = df[all_features]
y = df[TARGET]
groups = df[GROUP]


print()
print("Unseen-gateway validation")
print("=========================")
print()

print(f"Rows: {len(df)}")
print(f"Unique gateways: {groups.nunique()}")
print()

print("Class distribution:")
print(y.value_counts().sort_index().to_string())


# ---------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------
#
# StratifiedGroupKFold ensures that the same gateway
# cannot appear in both training and validation.
#
# Since each gateway occurs only once here, this is
# effectively a gateway-held-out evaluation.
# ---------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def evaluate(
    name,
    categorical_features
):

    features = (
        NUMERIC_FEATURES
        + categorical_features
    )

    model = make_model(
        categorical_features
    )

    probabilities = cross_val_predict(
        model,
        df[features],
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced = balanced_accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print(name)
    print("-" * len(name))

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Balanced accuracy: {balanced:.3f}"
    )

    print(
        f"ROC-AUC: {auc:.3f}"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    return {
        "model": name,
        "features": len(features),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
    }


# ---------------------------------------------------------
# Model A
# ---------------------------------------------------------

result_a = evaluate(
    "RF - current metadata",
    CURRENT_CATEGORICAL
)


# ---------------------------------------------------------
# Model B
# ---------------------------------------------------------

result_b = evaluate(
    "RF - all metadata",
    CURRENT_CATEGORICAL
    + EXTRA_CATEGORICAL
)


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------

comparison = pd.DataFrame([
    result_a,
    result_b
])

print()
print("Comparison")
print("==========")
print()

print(
    comparison.to_string(
        index=False
    )
)