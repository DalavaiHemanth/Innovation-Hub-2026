import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("ml_training.csv")

NUMERIC_FEATURES = [
    "7d_disconnection_cnt_mean", "14d_disconnection_cnt_mean", "28d_disconnection_cnt_mean",
    "7d_disconnection_cnt_max", "14d_disconnection_cnt_max", "28d_disconnection_cnt_max",
    "7d_disconnection_cnt_active_days", "14d_disconnection_cnt_active_days", "28d_disconnection_cnt_active_days",
    "7d_offline_duration_sec_mean", "14d_offline_duration_sec_mean", "28d_offline_duration_sec_mean",
    "7d_offline_duration_sec_max", "14d_offline_duration_sec_max", "28d_offline_duration_sec_max",
    "7d_avg_load1_mean", "14d_avg_load1_mean", "28d_avg_load1_mean",
    "7d_avg_load1_max", "14d_avg_load1_max", "28d_avg_load1_max",
    "7d_avg_uptime_mean", "14d_avg_uptime_mean", "28d_avg_uptime_mean",
    "7d_rssi_bad_ratio", "14d_rssi_bad_ratio", "28d_rssi_bad_ratio",
    "7d_rscp_bad_ratio", "14d_rscp_bad_ratio", "28d_rscp_bad_ratio",
    "7d_ecio_bad_ratio", "14d_ecio_bad_ratio", "28d_ecio_bad_ratio",
    "7d_rx_crc_bad_mean", "14d_rx_crc_bad_mean", "28d_rx_crc_bad_mean",
    "7d_tx_success_mean", "14d_tx_success_mean", "28d_tx_success_mean",
    "7d_reboot_cnt_mean", "14d_reboot_cnt_mean", "28d_reboot_cnt_mean",
    "meter_read_rate_mean", "meter_read_rate_std", "zero_read_weeks",
    "offline_duration_sec_recent_change", "reboot_cnt_recent_change",
]

CATEGORICAL_FEATURES = [
    "hw_model",
    "site_type",
    "region",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]
y = df["label"]
groups = df["gid"]


def make_model():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf),
    ])


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

all_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    test = df.iloc[test_idx][
        ["gid", "label"]
    ].copy()

    test["probability"] = probabilities

    # -----------------------------------------------------
    # Top 15 from this held-out fold
    # -----------------------------------------------------

    top15 = test.sort_values(
        "probability",
        ascending=False
    ).head(15)

    hits = int(top15["label"].sum())

    precision = hits / 15

    total_bad = int(y_test.sum())

    recall = (
        hits / total_bad
        if total_bad > 0
        else 0
    )

    all_results.append({
        "fold": fold,
        "test_gateways": len(test),
        "bad_gateways": total_bad,
        "top15_hits": hits,
        "precision_at_15": precision,
        "recall_at_15": recall,
    })

    print()
    print(f"Fold {fold}")
    print("-" * 20)
    print(f"Test gateways: {len(test)}")
    print(f"Schlecht gateways: {total_bad}")
    print(f"Top-15 Schlecht hits: {hits}")
    print(f"Precision@15: {precision:.3f}")
    print(f"Recall@15: {recall:.3f}")

    print()
    print("Top 15:")
    print(
        top15[
            ["gid", "label", "probability"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# Overall
# ---------------------------------------------------------

results = pd.DataFrame(all_results)

print()
print("=" * 60)
print("TOP-15 VALIDATION SUMMARY")
print("=" * 60)

print()
print(results.to_string(index=False))

print()
print("Average:")
print(
    f"Top-15 hits:       "
    f"{results.top15_hits.mean():.2f} / 15"
)

print(
    f"Precision@15:      "
    f"{results.precision_at_15.mean():.3f}"
)

print(
    f"Recall@15:         "
    f"{results.recall_at_15.mean():.3f}"
)

print()
print(
    f"Total top-15 hits:  "
    f"{results.top15_hits.sum()}"
)