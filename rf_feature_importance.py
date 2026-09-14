import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("ml_training.csv")

numeric = [
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
    "7d_avg_load1_mean",
    "14d_avg_load1_mean",
    "28d_avg_load1_mean",
    "7d_avg_load1_max",
    "28d_avg_load1_max",
    "7d_avg_memfree_mean",
    "28d_avg_memfree_mean",
    "7d_rssi_bad_ratio",
    "14d_rssi_bad_ratio",
    "28d_rssi_bad_ratio",
    "7d_rscp_bad_ratio",
    "14d_rscp_bad_ratio",
    "28d_rscp_bad_ratio",
    "7d_ecio_bad_ratio",
    "14d_ecio_bad_ratio",
    "28d_ecio_bad_ratio",
    "7d_tx_success_mean",
    "28d_tx_success_mean",
    "7d_tx_busy_mean",
    "28d_tx_busy_mean",
    "meter_read_rate_mean",
    "meter_read_rate_min",
    "meter_read_rate_std",
    "zero_read_weeks",
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

X = df[numeric + categorical]
y = df["label"]

preprocess = ColumnTransformer(
    [
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical,
        ),
    ]
)

model = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X, y)

names = model.named_steps[
    "preprocess"
].get_feature_names_out()

importance = model.named_steps[
    "classifier"
].feature_importances_

result = (
    pd.DataFrame(
        {
            "feature": names,
            "importance": importance,
        }
    )
    .sort_values("importance", ascending=False)
)

print("Top 25 Random Forest features")
print("-----------------------------")
print(result.head(25).to_string(index=False))