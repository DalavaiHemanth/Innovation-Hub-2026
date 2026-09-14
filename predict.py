import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


TELEMETRY_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "avg_reboot_duration",
    "avg_load1",
    "avg_memfree",
    "avg_uptime",
    "avg_activeproccess",
    "avg_totalproccess",
    "tx_success",
    "tx_busy",
    "tx_override",
    "rx_nr_pkts",
    "rx_crc_bad",
    "number_of_messages",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
    "rscp_rsrp_good",
    "rscp_rsrp_normal",
    "rscp_rsrp_bad",
    "ecio_rsrq_good",
    "ecio_rsrq_normal",
    "ecio_rsrq_bad",
]


# EXACT feature set used by train_random_forest.py
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


PREDICTION_WEEKS = [
    "2026-02-02",
    "2026-02-09",
    "2026-02-16",
    "2026-02-23",
    "2026-03-02",
    "2026-03-09",
    "2026-03-16",
    "2026-03-23",
]


def normalize_id(s):
    return s.astype(str).str.replace(":", "", regex=False).str.upper()


def safe_ratio(a, b):
    return a / b.replace(0, np.nan)


def load_telemetry(data_dir):
    print("Loading telemetry...")

    telemetry = pd.read_parquet(
        data_dir / "telemetry",
        columns=[
            "gateway_id",
            "ts_utc",
        ] + TELEMETRY_METRICS,
    )

    telemetry["gid"] = normalize_id(telemetry["gateway_id"])
    telemetry["ts"] = pd.to_datetime(telemetry["ts_utc"], utc=True)
    telemetry["date"] = telemetry["ts"].dt.floor("D")

    print(f"Telemetry rows: {len(telemetry):,}")

    return telemetry


def load_meter_data(data_dir):
    print("Loading meter-read data...")

    m = pd.read_csv(data_dir / "meter_read_success.csv")

    m["gid"] = normalize_id(m["gateway_id"])
    m["week_start"] = pd.to_datetime(m["week_start"])

    m["read_rate"] = safe_ratio(
        m["meters_read"],
        m["meters_expected"],
    )

    return m


def load_metadata(data_dir):
    print("Loading gateway metadata...")

    m = pd.read_csv(
        data_dir / "gateway_master.csv",
        encoding="cp1252",
    )

    m["gid"] = normalize_id(m["gateway_id"])

    return m[
        [
            "gid",
            "site_type",
            "hw_model",
            "region",
            "n_meters_installed",
        ]
    ].copy()


def build_features_for_week(
    telemetry,
    meter_data,
    metadata,
    week_start,
):
    """
    Build features using ONLY information strictly before week_start.
    """

    week_start = pd.Timestamp(
        week_start,
        tz="UTC",
    )

    # -----------------------------------------------------
    # TELEMETRY
    # -----------------------------------------------------

    t = telemetry[
        telemetry["ts"] < week_start
    ].copy()

    daily = (
        t.groupby(["gid", "date"])[TELEMETRY_METRICS]
        .mean()
        .reset_index()
    )

    problem_cols = [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
    ]

    for col in problem_cols:
        daily[f"{col}_active"] = (
            daily[col].fillna(0) > 0
        ).astype(int)

    daily["rssi_bad_ratio"] = safe_ratio(
        daily["rssi_bad"],
        daily["rssi_good"]
        + daily["rssi_normal"]
        + daily["rssi_bad"],
    )

    daily["rscp_bad_ratio"] = safe_ratio(
        daily["rscp_rsrp_bad"],
        daily["rscp_rsrp_good"]
        + daily["rscp_rsrp_normal"]
        + daily["rscp_rsrp_bad"],
    )

    daily["ecio_bad_ratio"] = safe_ratio(
        daily["ecio_rsrq_bad"],
        daily["ecio_rsrq_good"]
        + daily["ecio_rsrq_normal"]
        + daily["ecio_rsrq_bad"],
    )

    feature_rows = []

    # -----------------------------------------------------
    # ONE ROW PER GATEWAY
    # -----------------------------------------------------

    for gid in metadata["gid"].unique():

        g = daily[
            daily["gid"] == gid
        ]

        row = {
            "gid": gid
        }

        # 7 / 14 / 28 day windows
        for name, days in {
            "7d": 7,
            "14d": 14,
            "28d": 28,
        }.items():

            start = week_start - pd.Timedelta(
                days=days
            )

            w = g[
                (g["date"] >= start)
                & (g["date"] < week_start)
            ]

            row[f"{name}_days_available"] = len(w)

            for col in TELEMETRY_METRICS:

                row[
                    f"{name}_{col}_mean"
                ] = w[col].mean()

                row[
                    f"{name}_{col}_std"
                ] = w[col].std()

                row[
                    f"{name}_{col}_max"
                ] = w[col].max()

            for col in problem_cols:

                row[
                    f"{name}_{col}_active_days"
                ] = w[
                    f"{col}_active"
                ].sum()

            row[
                f"{name}_rssi_bad_ratio"
            ] = w[
                "rssi_bad_ratio"
            ].mean()

            row[
                f"{name}_rscp_bad_ratio"
            ] = w[
                "rscp_bad_ratio"
            ].mean()

            row[
                f"{name}_ecio_bad_ratio"
            ] = w[
                "ecio_bad_ratio"
            ].mean()

        # -------------------------------------------------
        # RECENT CHANGE FEATURES
        # -------------------------------------------------

        for col in [
            "offline_duration_sec",
            "disconnection_cnt",
            "reboot_cnt",
        ]:

            recent = row.get(
                f"7d_{col}_mean",
                np.nan,
            )

            baseline = row.get(
                f"28d_{col}_mean",
                np.nan,
            )

            if (
                pd.notna(baseline)
                and baseline != 0
            ):
                row[
                    f"{col}_recent_change"
                ] = (
                    recent - baseline
                ) / abs(baseline)

            else:
                row[
                    f"{col}_recent_change"
                ] = np.nan

        feature_rows.append(row)

    telemetry_features = pd.DataFrame(
        feature_rows
    )

    # -----------------------------------------------------
    # METER FEATURES
    # -----------------------------------------------------

    meter_cutoff = week_start.tz_localize(None)

    m = meter_data[
        meter_data["week_start"]
        < meter_cutoff
    ]

    meter_features = (
        m.groupby("gid")
        .agg(
            meter_read_rate_mean=(
                "read_rate",
                "mean",
            ),
            meter_read_rate_std=(
                "read_rate",
                "std",
            ),
            zero_read_weeks=(
                "read_rate",
                lambda x: (x == 0).sum(),
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    result = metadata.merge(
        telemetry_features,
        on="gid",
        how="left",
    )

    result = result.merge(
        meter_features,
        on="gid",
        how="left",
    )

    return result


def make_reason(row):
    """
    Generate a human-readable explanation
    from the strongest available risk signals.
    """

    reasons = []

    disconnections = row.get(
        "7d_disconnection_cnt_mean",
        np.nan,
    )

    offline = row.get(
        "7d_offline_duration_sec_mean",
        np.nan,
    )

    load = row.get(
        "7d_avg_load1_mean",
        np.nan,
    )

    rssi = row.get(
        "7d_rssi_bad_ratio",
        np.nan,
    )

    reboot = row.get(
        "7d_reboot_cnt_mean",
        np.nan,
    )

    meter = row.get(
        "meter_read_rate_mean",
        np.nan,
    )

    if pd.notna(disconnections) and disconnections > 0:
        reasons.append("recent disconnections")

    if pd.notna(offline) and offline > 0:
        reasons.append("offline duration")

    if pd.notna(load) and load > 1:
        reasons.append("elevated system load")

    if pd.notna(rssi) and rssi > 0.20:
        reasons.append("poor signal quality")

    if pd.notna(reboot) and reboot > 0:
        reasons.append("recent reboots")

    if (
        pd.notna(meter)
        and meter < 0.80
    ):
        reasons.append("low meter-read rate")

    if not reasons:
        reasons.append(
            "model-ranked based on combined gateway telemetry"
        )

    return "; ".join(reasons[:3])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data",
    )

    parser.add_argument(
        "--model",
        default="model_rf.pkl",
    )

    parser.add_argument(
        "--out",
        default="predictions.csv",
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    print("Loading model...")
    model = joblib.load(args.model)

    telemetry = load_telemetry(data_dir)
    meter_data = load_meter_data(data_dir)
    metadata = load_metadata(data_dir)

    all_predictions = []

    print()
    print("Generating weekly predictions")
    print("-----------------------------")

    for week in PREDICTION_WEEKS:

        print(f"Processing {week}...")

        features = build_features_for_week(
            telemetry,
            meter_data,
            metadata,
            week,
        )

        # Make absolutely sure prediction columns
        # match the training model.
        X = features[
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        ]

        probabilities = model.predict_proba(X)[:, 1]

        features["score"] = probabilities

        # Highest risk first.
        ranked = features.sort_values(
            "score",
            ascending=False,
        ).head(15).copy()

        ranked["week_start"] = week

        ranked["rank"] = range(
            1,
            len(ranked) + 1,
        )

        ranked["reason"] = ranked.apply(
            make_reason,
            axis=1,
        )

        output = ranked[
            [
                "week_start",
                "rank",
                "gid",
                "score",
                "reason",
            ]
        ].rename(
            columns={
                "gid": "gateway_id"
            }
        )

        all_predictions.append(output)

        print(
            f"  Selected: {len(output)} gateways"
        )

    result = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    result.to_csv(
        args.out,
        index=False,
    )

    print()
    print("-----------------------------")
    print("Prediction complete")
    print("-----------------------------")
    print(f"Rows: {len(result)}")
    print(f"Weeks: {result['week_start'].nunique()}")
    print(f"Output: {args.out}")

    print()
    print("Rows per week:")
    print(
        result.groupby("week_start")
        .size()
        .to_string()
    )

    print()
    print("Top selected gateways:")
    print(
        result["gateway_id"]
        .value_counts()
        .head(15)
        .to_string()
    )


if __name__ == "__main__":
    main()