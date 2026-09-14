import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# ---------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------

df = pd.read_csv("ml_training.csv")

target = "label"

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

categorical_features = ["hw_model", "site_type", "region"]

# Keep only columns that actually exist.
numeric_features = [
    c for c in numeric_features
    if c in df.columns
]

categorical_features = [
    c for c in categorical_features
    if c in df.columns
]

X = df[numeric_features + categorical_features]
y = df[target]


# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])


# ---------------------------------------------------------
# 4. Cross-validation
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

probs = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

predictions = (probs >= 0.5).astype(int)

print("Random Forest training")
print("----------------------")
print(f"Rows: {len(df)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features: {len(numeric_features) + len(categorical_features)}")

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\n5-fold cross-validation")
print("-----------------------")
print(f"Accuracy: {accuracy_score(y, predictions):.3f}")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y, predictions):.3f}"
)
print(f"ROC-AUC: {roc_auc_score(y, probs):.3f}")


# ---------------------------------------------------------
# 5. Fit final model on ALL reviewed gateways
# ---------------------------------------------------------

model.fit(X, y)

joblib.dump(model, "model_rf.pkl")

print("\nFinal model")
print("-----------")
print("Saved: model_rf.pkl")