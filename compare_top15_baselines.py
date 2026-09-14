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

CATEGORICAL_FEATURES = [
    "hw_model",
    "site_type",
    "region",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]
y = df["label"]
groups = df["gid"]


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

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
        ("cat", categorical, CATEGORICAL_FEATURES)
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
# Composite operational score
# ---------------------------------------------------------

def percentile_rank(series):

    return series.rank(
        pct=True,
        method="average"
    ).fillna(0)


def composite_score(test):

    # Higher = worse
    disconnection = percentile_rank(
        test["28d_disconnection_cnt_mean"]
    )

    offline = percentile_rank(
        test["28d_offline_duration_sec_mean"]
    )

    load = percentile_rank(
        test["28d_avg_load1_mean"]
    )

    signal = percentile_rank(
        test["28d_rssi_bad_ratio"]
    )

    reboot = percentile_rank(
        test["28d_reboot_cnt_mean"]
    )

    return (
        0.35 * disconnection
        + 0.25 * offline
        + 0.20 * load
        + 0.10 * signal
        + 0.10 * reboot
    )


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    # -----------------------------------------------------
    # RF
    # -----------------------------------------------------

    model = make_model()

    model.fit(
        train[FEATURES],
        train["label"]
    )

    test["rf_score"] = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    # -----------------------------------------------------
    # Disconnection-only
    # -----------------------------------------------------

    test["disconnect_score"] = (
        test["28d_disconnection_cnt_mean"]
    )

    # -----------------------------------------------------
    # Composite score
    # -----------------------------------------------------

    test["composite_score"] = composite_score(
        test
    )

    # -----------------------------------------------------
    # Evaluate each ranking
    # -----------------------------------------------------

    rankings = {
        "Disconnection only":
            "disconnect_score",

        "Composite health":
            "composite_score",

        "Random Forest":
            "rf_score",
    }

    print()
    print(f"Fold {fold}")
    print("=" * 50)

    for name, score_col in rankings.items():

        top15 = test.sort_values(
            score_col,
            ascending=False
        ).head(15)

        hits = int(
            top15["label"].sum()
        )

        bad_total = int(
            test["label"].sum()
        )

        precision = hits / 15

        recall = (
            hits / bad_total
            if bad_total > 0
            else 0
        )

        results.append({
            "fold": fold,
            "method": name,
            "hits": hits,
            "precision": precision,
            "recall": recall,
        })

        print(
            f"{name:22s} "
            f"hits={hits:2d}/15 "
            f"precision={precision:.3f} "
            f"recall={recall:.3f}"
        )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("method")
    .agg(
        average_hits=("hits", "mean"),
        total_hits=("hits", "sum"),
        precision_at_15=("precision", "mean"),
        recall_at_15=("recall", "mean"),
    )
    .sort_values(
        "recall_at_15",
        ascending=False
    )
)


print()
print("=" * 65)
print("TOP-15 METHOD COMPARISON")
print("=" * 65)

print()
print(summary.round(3).to_string())

print()
print(
    "Interpretation: higher Recall@15 means more "
    "Schlecht gateways captured within the 15 available visits."
)