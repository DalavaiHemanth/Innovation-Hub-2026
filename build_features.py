import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CUTOFF = pd.Timestamp("2026-02-15", tz="UTC")

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


def normalize_id(s):
    return s.astype(str).str.replace(":", "", regex=False).str.upper()


def safe_ratio(a, b):
    return a / b.replace(0, np.nan)


def build_telemetry_features(telemetry):
    telemetry["gid"] = normalize_id(telemetry["gateway_id"])
    telemetry["ts"] = pd.to_datetime(telemetry["ts_utc"])

    # Only information available before the engineer review.
    telemetry = telemetry[telemetry["ts"] < CUTOFF].copy()

    # Convert to date for rolling windows.
    telemetry["date"] = telemetry["ts"].dt.floor("D")

    # Aggregate hourly telemetry to gateway/day.
    daily = (
        telemetry.groupby(["gid", "date"])[TELEMETRY_METRICS]
        .mean()
        .reset_index()
    )

    # Number of hours with operational problems.
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
        daily["rssi_good"] + daily["rssi_normal"] + daily["rssi_bad"],
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

    # Most recent day represented in the data.
    latest = daily["date"].max()

    feature_rows = []

    for gid, g in daily.groupby("gid"):
        row = {"gid": gid}

        # Windows are measured backwards from the review cutoff.
        windows = {
            "7d": 7,
            "14d": 14,
            "28d": 28,
        }

        for name, days in windows.items():
            start = CUTOFF - pd.Timedelta(days=days)

            w = g[
                (g["date"] >= start)
                & (g["date"] < CUTOFF)
            ]

            row[f"{name}_days_available"] = len(w)

            # Core telemetry statistics.
            for col in TELEMETRY_METRICS:
                row[f"{name}_{col}_mean"] = w[col].mean()
                row[f"{name}_{col}_std"] = w[col].std()
                row[f"{name}_{col}_max"] = w[col].max()

            # Operational-event counts.
            for col in problem_cols:
                row[f"{name}_{col}_active_days"] = (
                    w[f"{col}_active"].sum()
                )

            # Signal quality.
            row[f"{name}_rssi_bad_ratio"] = w["rssi_bad_ratio"].mean()
            row[f"{name}_rscp_bad_ratio"] = w["rscp_bad_ratio"].mean()
            row[f"{name}_ecio_bad_ratio"] = w["ecio_bad_ratio"].mean()

        # Recent-vs-long-term changes.
        for col in [
            "offline_duration_sec",
            "disconnection_cnt",
            "reboot_cnt",
            "avg_load1",
            "avg_uptime",
            "rssi_bad_ratio",
        ]:
            recent = row.get(f"7d_{col}_mean", np.nan)
            baseline = row.get(f"28d_{col}_mean", np.nan)

            if pd.notna(baseline) and baseline != 0:
                row[f"{col}_recent_change"] = (
                    recent - baseline
                ) / abs(baseline)
            else:
                row[f"{col}_recent_change"] = np.nan

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def build_meter_features(data_dir):
    path = data_dir / "meter_read_success.csv"

    m = pd.read_csv(path)
    m["gid"] = normalize_id(m["gateway_id"])
    m["week_start"] = pd.to_datetime(m["week_start"])
    meter_cutoff = CUTOFF.tz_localize(None)

    m = m[m["week_start"] < meter_cutoff].copy()



    m["read_rate"] = safe_ratio(
        m["meters_read"],
        m["meters_expected"],
    )

    features = (
        m.groupby("gid")
        .agg(
            meter_read_rate_mean=("read_rate", "mean"),
            meter_read_rate_min=("read_rate", "min"),
            meter_read_rate_std=("read_rate", "std"),
            meter_weeks_available=("read_rate", "size"),
            zero_read_weeks=(
                "read_rate",
                lambda x: (x == 0).sum(),
            ),
        )
        .reset_index()
    )

    return features


def build_metadata_features(data_dir):
    path = data_dir / "gateway_master.csv"

    m = pd.read_csv(path, encoding="cp1252")
    m["gid"] = normalize_id(m["gateway_id"])

    cols = [
        "gid",
        "site_type",
        "hw_model",
        "antenna_type",
        "fw_version",
        "region",
        "n_meters_installed",
    ]

    return m[cols].copy()


def build_labels(data_dir):
    path = data_dir / "engineer_review_2026-02.xlsx"

    r = pd.read_excel(path)
    r["gid"] = normalize_id(r["gateway_id"])
    r["label"] = (r["Kategorie"] == "Schlecht").astype(int)

    return r[["gid", "Kategorie", "label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="ml_training.csv")
    args = parser.parse_args()

    data_dir = Path(args.data)

    print("Loading telemetry...")
    telemetry = pd.read_parquet(
        data_dir / "telemetry",
        columns=["gateway_id", "ts_utc"] + TELEMETRY_METRICS,
    )

    print(f"Telemetry rows: {len(telemetry):,}")

    print("Building telemetry features...")
    telemetry_features = build_telemetry_features(telemetry)

    print("Building meter-read features...")
    meter_features = build_meter_features(data_dir)

    print("Loading gateway metadata...")
    metadata = build_metadata_features(data_dir)

    print("Loading engineer labels...")
    labels = build_labels(data_dir)

    # Start from the labelled gateways.
    result = labels.merge(
        telemetry_features,
        on="gid",
        how="left",
    )

    result = result.merge(
        meter_features,
        on="gid",
        how="left",
    )

    result = result.merge(
        metadata,
        on="gid",
        how="left",
    )

    result.to_csv(args.out, index=False)

    print()
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print()
    print("Labels:")
    print(result["Kategorie"].value_counts().to_string())
    print()
    print("Telemetry availability:")
    print(
        result["28d_days_available"]
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()