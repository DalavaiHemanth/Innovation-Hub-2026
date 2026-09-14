import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

X = df[numeric + categorical]
y = df["label"]

preprocess = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
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
            LogisticRegression(
                max_iter=2000,
                C=0.1,
                class_weight="balanced",
            ),
        ),
    ]
)

model.fit(X, y)

feature_names = model.named_steps[
    "preprocess"
].get_feature_names_out()

coefficients = model.named_steps[
    "classifier"
].coef_[0]

importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }
).sort_values(
    "abs_coefficient",
    ascending=False,
)

print("Top features pushing toward Schlecht:")
print(
    importance.head(20).to_string(index=False)
)

print("\nTop features pushing toward Normal:")
print(
    importance.tail(20)
    .sort_values("coefficient")
    .to_string(index=False)
)