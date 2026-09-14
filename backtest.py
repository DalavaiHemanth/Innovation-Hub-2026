import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")

PREDICTION_WEEKS = [
    "2025-10-06",
    "2025-10-13",
    "2025-10-20",
    "2025-10-27",
    "2025-11-03",
    "2025-11-10",
    "2025-11-17",
    "2025-11-24",
    "2025-12-01",
    "2025-12-08",
    "2025-12-15",
    "2025-12-22",
    "2025-12-29",
    "2026-01-05",
    "2026-01-12",
    "2026-01-19",
]


def normalize_id(s):
    return (
        s.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )


# ---------------------------------------------------------
# Load field visits
# ---------------------------------------------------------

visits = pd.read_csv(
    DATA_DIR / "field_visits.csv"
)

visits["gateway_id"] = normalize_id(
    visits["gateway_id"]
)

visits["requested_on"] = pd.to_datetime(
    visits["requested_on"]
)

visits["visited_on"] = pd.to_datetime(
    visits["visited_on"]
)


# ---------------------------------------------------------
# Load meter reads
# ---------------------------------------------------------

meter = pd.read_csv(
    DATA_DIR / "meter_read_success.csv"
)

meter["gateway_id"] = normalize_id(
    meter["gateway_id"]
)

meter["week_start"] = pd.to_datetime(
    meter["week_start"]
)

meter["read_rate"] = (
    meter["meters_read"]
    / meter["meters_expected"].replace(0, np.nan)
)


# ---------------------------------------------------------
# Load ML predictions
# ---------------------------------------------------------

ml = pd.read_csv(
    "predictions_ml.csv"
)

ml["week_start"] = pd.to_datetime(
    ml["week_start"]
)

ml["gateway_id"] = normalize_id(
    ml["gateway_id"]
)


# ---------------------------------------------------------
# Load baseline predictions
# ---------------------------------------------------------

baseline = pd.read_csv(
    "predictions_baseline.csv"
)

baseline["week_start"] = pd.to_datetime(
    baseline["week_start"]
)

baseline["gateway_id"] = normalize_id(
    baseline["gateway_id"]
)


results = []


for week in PREDICTION_WEEKS:

    week = pd.Timestamp(week)

    # -----------------------------------------------------
    # Outcome window
    #
    # Look at field visits requested during the following
    # week. This avoids using visits that were requested
    # before our prediction decision.
    # -----------------------------------------------------

    next_week = week + pd.Timedelta(days=7)

    future_visits = visits[
        (visits["requested_on"] >= week)
        & (visits["requested_on"] < next_week)
    ].copy()

    fixed_gateways = set(
        future_visits.loc[
            future_visits["outcome"] == "Fehler behoben",
            "gateway_id",
        ]
    )

    no_error_gateways = set(
        future_visits.loc[
            future_visits["outcome"] == "Kein Fehler gefunden",
            "gateway_id",
        ]
    )

    no_access_gateways = set(
        future_visits.loc[
            future_visits["outcome"] == "Kein Zugang",
            "gateway_id",
        ]
    )

    # -----------------------------------------------------
    # Meter-read failures during the prediction week
    # -----------------------------------------------------

    weekly_meter = meter[
        meter["week_start"] == week
    ].copy()

    low_read_gateways = set(
        weekly_meter.loc[
            weekly_meter["read_rate"] < 0.50,
            "gateway_id",
        ]
    )

    # -----------------------------------------------------
    # Evaluate ML
    # -----------------------------------------------------

    ml_week = ml[
        ml["week_start"] == week
    ]

    ml_selected = set(
        ml_week["gateway_id"]
    )

    ml_fixed = len(
        ml_selected & fixed_gateways
    )

    ml_no_error = len(
        ml_selected & no_error_gateways
    )

    ml_low_read = len(
        ml_selected & low_read_gateways
    )

    # -----------------------------------------------------
    # Evaluate baseline
    # -----------------------------------------------------

    baseline_week = baseline[
        baseline["week_start"] == week
    ]

    baseline_selected = set(
        baseline_week["gateway_id"]
    )

    baseline_fixed = len(
        baseline_selected & fixed_gateways
    )

    baseline_no_error = len(
        baseline_selected & no_error_gateways
    )

    baseline_low_read = len(
        baseline_selected & low_read_gateways
    )

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    results.append({
        "week": week.date(),

        "future_field_visits":
            len(future_visits),

        "confirmed_fixes":
            len(fixed_gateways),

        "no_error_visits":
            len(no_error_gateways),

        "no_access_visits":
            len(no_access_gateways),

        "low_read_gateways":
            len(low_read_gateways),

        "ml_fix_hits":
            ml_fixed,

        "ml_no_error_hits":
            ml_no_error,

        "ml_low_read_hits":
            ml_low_read,

        "baseline_fix_hits":
            baseline_fixed,

        "baseline_no_error_hits":
            baseline_no_error,

        "baseline_low_read_hits":
            baseline_low_read,
    })


result = pd.DataFrame(results)


# ---------------------------------------------------------
# Print weekly results
# ---------------------------------------------------------

print()
print("Historical operational backtest")
print("================================")
print()

print(
    result.to_string(index=False)
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("Summary")
print("=======")

print(
    f"ML confirmed fixes hit: "
    f"{result.ml_fix_hits.sum()}"
)

print(
    f"Baseline confirmed fixes hit: "
    f"{result.baseline_fix_hits.sum()}"
)

print()

print(
    f"ML no-error visits: "
    f"{result.ml_no_error_hits.sum()}"
)

print(
    f"Baseline no-error visits: "
    f"{result.baseline_no_error_hits.sum()}"
)

print()

print(
    f"ML low-read hits: "
    f"{result.ml_low_read_hits.sum()}"
)

print(
    f"Baseline low-read hits: "
    f"{result.baseline_low_read_hits.sum()}"
)

print()

print(
    "Total confirmed fixes available: "
    f"{result.confirmed_fixes.sum()}"
)