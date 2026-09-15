============================================================
NEXORA 2026 - FINAL ML SUBMISSION
LPDG INNOVATION HUB - SELECTION CHALLENGE 2026
============================================================

PROJECT OVERVIEW

This submission uses Machine Learning for the Part 2 challenge.

The final solution uses a Random Forest model to rank gateways by
failure risk and select the 15 gateways that should receive field
attention each week.

The solution is designed to operate using only information available
before each prediction week.

------------------------------------------------------------
QUICK START
------------------------------------------------------------

1. Install Python dependencies:

    pip install -r requirements.txt

2. Place the challenge data in:

    data/

3. Run the complete pipeline:

    python run.py

The pipeline automatically:

    1. Builds the ML training features.
    2. Trains the Random Forest model.
    3. Generates predictions for all required weeks.
    4. Validates predictions.csv.

Final output:

    predictions.csv

The validator confirms:

    15 ranked gateways for each of 8 weeks
    120 total prediction rows

------------------------------------------------------------
PIPELINE
------------------------------------------------------------

The complete pipeline is:

    data/
       |
       v
    build_features.py
       |
       v
    ml_training.csv
       |
       v
    train_random_forest.py
       |
       v
    model_rf.pkl
       |
       v
    predict.py
       |
       v
    predictions.csv
       |
       v
    validate_submission.py

run.py orchestrates these steps so the complete solution can
be executed with one command.

The training and prediction stages remain separate:

    build_features.py + train_random_forest.py
        = training

    predict.py
        = prediction

This keeps model training separate from weekly prediction.

------------------------------------------------------------
MODEL
------------------------------------------------------------

Model:

    Random Forest

Final model features:

    44 numeric features
    3 categorical features

Categorical features:

    hw_model
    site_type
    region

Gateway ID is deliberately excluded from the model.

The model uses telemetry-derived information including:

    - disconnections
    - offline duration
    - reboots
    - load
    - uptime
    - signal quality
    - packet/CRC-related metrics
    - transmission metrics
    - meter-read success
    - 7, 14 and 28 day statistics
    - recent-vs-history changes

------------------------------------------------------------
VALIDATION
------------------------------------------------------------

Five-fold cross-validation:

    Accuracy:          0.767
    Balanced accuracy: 0.767
    ROC-AUC:           0.848

Gateway-held-out validation:

    Accuracy:          0.792
    Balanced accuracy: 0.792
    ROC-AUC:           0.850

Top-15 validation:

    Average useful gateways: 10.8 / 15
    Precision@15:             0.720
    Recall:                   0.905

The top-15 results are validation results on the labelled
engineer-review dataset.

A complete eight-week operational euro-cost comparison is not
claimed because the available historical field-visit records do
not cover the complete scored period.

------------------------------------------------------------
SUBMISSION FILES
------------------------------------------------------------

Important files:

    run.py
        One-command pipeline runner.

    build_features.py
        Builds the supervised ML training dataset.

    train_random_forest.py
        Trains and saves the Random Forest model.

    predict.py
        Generates weekly gateway rankings.

    validate_submission.py
        Validates the required predictions.csv format.

    predictions.csv
        Final predictions for the eight scored weeks.

    DECISIONS.md
        Modelling decisions, alternatives, validation and limitations.

    AI-USAGE.md
        AI assistance and human verification record.

    requirements.txt
        Python dependencies.

The original challenge files are also included:

    01-Challenge-Brief.pdf
    02-Data-Dictionary.pdf
    03-challenge-data.zip
    baseline_3sigma.py

------------------------------------------------------------
ORIGINAL CHALLENGE QUICK START
------------------------------------------------------------

The supplied baseline can still be run independently with:

    python baseline_3sigma.py --data data --out predictions.csv

However, the final submission uses the ML pipeline:

    python run.py

------------------------------------------------------------
LIVE SESSION
------------------------------------------------------------

If invited to a live session, the model can be retrained and
the prediction pipeline rerun after receiving new data.

The model is intended as a prioritization system rather than a
guarantee that every selected gateway will fail.

------------------------------------------------------------
DATA
------------------------------------------------------------

The challenge data is for this exercise only.

Do not pass the data on or publish it.

------------------------------------------------------------
DOCUMENTATION
------------------------------------------------------------

For detailed modelling decisions:

    DECISIONS.md

For AI usage and verification:

    AI-USAGE.md

============================================================