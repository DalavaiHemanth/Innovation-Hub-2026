import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

VISIT_COST = 380
MISSED_WEEK_COST = 600

PREDICTION_FILE = "predictions_ml.csv"
BASELINE_FILE = "predictions_baseline.csv"
VISITS_FILE = "data/field_visits.csv"


# ============================================================
# Load data
# ============================================================

ml = pd.read_csv(PREDICTION_FILE)
baseline = pd.read_csv(BASELINE_FILE)
visits = pd.read_csv(VISITS_FILE)


# ============================================================
# Normalize gateway IDs
# ============================================================

def normalize_gid(x):
    if pd.isna(x):
        return None

    x = str(x).strip().upper()

    # Handle IDs that may appear as colon-separated values
    x = x.replace(":", "")

    return x


ml["gateway_id"] = ml["gateway_id"].apply(normalize_gid)
baseline["gateway_id"] = baseline["gateway_id"].apply(normalize_gid)


# ============================================================
# Inspect field-visit data
# ============================================================

print("\nField visit columns:")
print(visits.columns.tolist())

# Actual columns in field_visits.csv
date_col = "visited_on"
gid_col = "gateway_id"
outcome_col = "outcome"


if date_col is None or gid_col is None or outcome_col is None:
    raise ValueError(
        f"Could not identify visit columns.\n"
        f"date={date_col}, gateway={gid_col}, outcome={outcome_col}"
    )


visits[date_col] = pd.to_datetime(
    visits[date_col],
    errors="coerce"
)

visits["gateway_id"] = visits[gid_col].apply(
    normalize_gid
)

visits["outcome_clean"] = (
    visits[outcome_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# Define actionable outcomes
# ============================================================

# Confirmed problem
visits["confirmed_problem"] = (
    visits["outcome_clean"]
    == "Fehler behoben"
)

# Actual wasted visit
visits["wasted_visit"] = (
    visits["outcome_clean"]
    == "Kein Fehler gefunden"
)


print("\nVisit outcome counts:")
print(
    visits["outcome_clean"]
    .value_counts(dropna=False)
)


# ============================================================
# Historical observed-value evaluation
# ============================================================

def evaluate_strategy(predictions, strategy_name):

    records = []

    for week in sorted(
        predictions["week_start"].unique()
    ):

        week_pred = predictions[
            predictions["week_start"] == week
        ].copy()

        selected = set(
            week_pred["gateway_id"]
        )

        # Look only at visits occurring during
        # the prediction week.
        week_date = pd.Timestamp(week)

        week_visits = visits[
            (visits[date_col] >= week_date)
            &
            (visits[date_col] < week_date + pd.Timedelta(days=7))
        ]

        selected_visits = week_visits[
            week_visits["gateway_id"].isin(selected)
        ]

        useful = int(
            selected_visits["confirmed_problem"].sum()
        )

        wasted = int(
            selected_visits["wasted_visit"].sum()
        )

        unknown = len(selected_visits) - useful - wasted

        observed_cost = (
            useful * 0
            + wasted * VISIT_COST
        )

        records.append({
            "week": week,
            "strategy": strategy_name,
            "selected": len(selected),
            "observed_visits": len(selected_visits),
            "useful_visits": useful,
            "wasted_visits": wasted,
            "unknown_visits": unknown,
            "observed_cost": observed_cost,
        })

    return pd.DataFrame(records)


# ============================================================
# Evaluate both strategies
# ============================================================

ml_results = evaluate_strategy(
    ml,
    "Random Forest"
)

baseline_results = evaluate_strategy(
    baseline,
    "3-Sigma baseline"
)


results = pd.concat(
    [ml_results, baseline_results],
    ignore_index=True
)


# ============================================================
# Print weekly results
# ============================================================

print("\n")
print("=" * 80)
print("HISTORICAL OBSERVED-VISIT COMPARISON")
print("=" * 80)

print(
    results[
        [
            "week",
            "strategy",
            "selected",
            "observed_visits",
            "useful_visits",
            "wasted_visits",
            "unknown_visits",
            "observed_cost",
        ]
    ].to_string(index=False)
)


# ============================================================
# Summary
# ============================================================

summary = (
    results
    .groupby("strategy")
    .agg(
        total_selected=("selected", "sum"),
        total_observed_visits=("observed_visits", "sum"),
        useful_visits=("useful_visits", "sum"),
        wasted_visits=("wasted_visits", "sum"),
        unknown_visits=("unknown_visits", "sum"),
        observed_cost=("observed_cost", "sum"),
    )
)

summary["useful_rate"] = (
    summary["useful_visits"]
    / summary["total_observed_visits"].replace(0, np.nan)
)

summary["wasted_rate"] = (
    summary["wasted_visits"]
    / summary["total_observed_visits"].replace(0, np.nan)
)

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    summary.round(3).to_string()
)


# ============================================================
# Data coverage
# ============================================================

print("\n")
print("=" * 80)
print("GROUND-TRUTH COVERAGE")
print("=" * 80)

print(
    f"Field visits available from "
    f"{visits[date_col].min().date()} "
    f"to {visits[date_col].max().date()}"
)

print(
    f"Challenge predictions from "
    f"{ml['week_start'].min()} "
    f"to {ml['week_start'].max()}"
)

print("""
The field-visit data does not cover the full challenge
prediction period.

Therefore the observed visit cost is only a partial
historical proxy and must NOT be presented as the
complete 8-week challenge cost.
""")