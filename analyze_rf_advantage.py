import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("ml_training.csv")

target = "label"


# ============================================================
# 2. Same features as train_random_forest.py
# ============================================================

numeric_features = [
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

    "7d_rscp_rsrp_bad_ratio",
    "14d_rscp_rsrp_bad_ratio",
    "28d_rscp_rsrp_bad_ratio",

    "7d_ecio_rsrq_bad_ratio",
    "14d_ecio_rsrq_bad_ratio",
    "28d_ecio_rsrq_bad_ratio",

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

categorical_features = [
    "hw_model",
    "site_type",
    "region",
    "firmware",
    "gateway_id",
]

# Keep only columns that actually exist.
numeric_features = [
    c for c in numeric_features
    if c in df.columns
]

categorical_features = [
    c for c in categorical_features
    if c in df.columns
]


FEATURES = numeric_features + categorical_features

X = df[FEATURES]
y = df[target]


# ============================================================
# 3. Model factory
# ============================================================

def make_model():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
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


# ============================================================
# 4. Helper: percentile ranking
# ============================================================

def percentile_rank(series):

    return series.rank(
        pct=True,
        method="average"
    ).fillna(0)


# ============================================================
# 5. Compare RF against disconnection ranking
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


all_advantages = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    model = make_model()

    model.fit(
        train[FEATURES],
        train[target]
    )

    # RF probability
    test["rf_score"] = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    # Simple baseline:
    # higher 28-day disconnection count = higher risk
    test["disconnect_score"] = (
        test["28d_disconnection_cnt_mean"]
    )

    # --------------------------------------------------------
    # Top 15 from each method
    # --------------------------------------------------------

    rf_top = set(
        test.nlargest(
            min(15, len(test)),
            "rf_score"
        )["gid"]
    )

    disconnect_top = set(
        test.nlargest(
            min(15, len(test)),
            "disconnect_score"
        )["gid"]
    )

    # Gateways RF catches but disconnection ranking misses
    rf_advantage = rf_top - disconnect_top

    # Gateways disconnection ranking catches but RF misses
    disconnect_advantage = (
        disconnect_top - rf_top
    )

    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print(
        f"RF Top-15: {len(rf_top)} gateways"
    )

    print(
        f"Disconnection Top-15: "
        f"{len(disconnect_top)} gateways"
    )

    print(
        f"RF advantage: "
        f"{len(rf_advantage)} gateways"
    )

    print(
        f"Disconnection advantage: "
        f"{len(disconnect_advantage)} gateways"
    )

    # --------------------------------------------------------
    # Show RF advantage cases
    # --------------------------------------------------------

    if len(rf_advantage) > 0:

        print("\nRF catches these gateways that")
        print("disconnection-only ranking misses:\n")

        cols = [
            "gid",
            "label",
            "rf_score",
            "disconnect_score",

            "28d_disconnection_cnt_mean",
            "28d_offline_duration_sec_mean",
            "28d_avg_load1_mean",
            "28d_rssi_bad_ratio",
            "28d_reboot_cnt_mean",

            "meter_read_rate_mean",
            "zero_read_weeks",

            "hw_model",
            "site_type",
            "region",
        ]

        cols = [
            c for c in cols
            if c in test.columns
        ]

        advantage_df = test[
            test["gid"].isin(rf_advantage)
        ][cols].sort_values(
            "rf_score",
            ascending=False
        )

        print(
            advantage_df.to_string(
                index=False
            )
        )

        advantage_df["fold"] = fold

        all_advantages.append(
            advantage_df
        )


# ============================================================
# 6. Combined RF advantage cases
# ============================================================

print()
print("=" * 70)
print("OVERALL RF ADVANTAGE")
print("=" * 70)

if all_advantages:

    combined = pd.concat(
        all_advantages,
        ignore_index=True
    )

    print(
        f"\nTotal RF advantage cases: "
        f"{len(combined)}"
    )

    print(
        f"Problematic gateways among them: "
        f"{combined['label'].sum()} "
        f"/ {len(combined)}"
    )

    print("\nAverage feature values:")

    feature_cols = [
        "28d_disconnection_cnt_mean",
        "28d_offline_duration_sec_mean",
        "28d_avg_load1_mean",
        "28d_rssi_bad_ratio",
        "28d_reboot_cnt_mean",
        "meter_read_rate_mean",
        "zero_read_weeks",
    ]

    feature_cols = [
        c for c in feature_cols
        if c in combined.columns
    ]

    print(
        combined[
            feature_cols
        ].mean().round(3).to_string()
    )

    print("\nFull RF advantage cases:")
    print(
        combined.to_string(
            index=False
        )
    )

else:

    print(
        "No RF advantage cases found."
    )