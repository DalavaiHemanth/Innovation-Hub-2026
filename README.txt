============================================================
NEXORA 2026 - FINAL ML SUBMISSION
LPDG INNOVATION HUB - SELECTION CHALLENGE 2026
============================================================

PROJECT OVERVIEW

This submission uses Machine Learning as the primary approach
for Part 2 of the NEXORA 2026 challenge.

The final solution uses a Random Forest model to estimate gateway
risk, rank gateways, and select the 15 gateways that should receive
field attention each week.

The prediction pipeline is designed to use only information that
is available before each prediction week.

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

The provided validator confirms:

    15 ranked gateways for each of 8 weeks
    120 total prediction rows

------------------------------------------------------------
PIPELINE
------------------------------------------------------------

The complete workflow is:

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

run.py orchestrates the complete workflow so it can be executed
with a single command.

The training and prediction stages remain separate:

    build_features.py + train_random_forest.py
                |
                v
             Training

    predict.py
        |
        v
      Prediction

This separation makes the workflow easier to reproduce and allows
the prediction stage to use a trained model with data available
before the prediction week.

------------------------------------------------------------
PART 2 - MACHINE LEARNING APPROACH
------------------------------------------------------------

WHY MACHINE LEARNING?

The challenge provides historical engineer-reviewed gateway labels
containing Normal and Schlecht examples. These labelled examples
allow the problem to be treated as a supervised classification task.

However, the operational requirement is not simply to classify every
gateway. The field team has a maximum capacity of 15 visits per week.

Therefore, the solution:

    1. Estimates risk for each gateway.
    2. Produces a risk score.
    3. Ranks gateways by that score.
    4. Selects the top 15 gateways for field attention each week.

Simpler decision-engineering approaches were also considered, but
Machine Learning was selected as the primary Part 2 approach.

------------------------------------------------------------
DATA AND FEATURE ENGINEERING
------------------------------------------------------------

The feature engineering pipeline combines:

    - Gateway telemetry
    - Meter-read success information
    - Selected gateway metadata

Telemetry is aggregated to gateway-level features over multiple
time windows:

    - 7 days
    - 14 days
    - 28 days

The features capture different aspects of gateway behaviour,
including:

    - Disconnections
    - Offline duration
    - Reboots
    - Load
    - Uptime
    - Signal quality
    - Packet/CRC-related metrics
    - Transmission-related metrics
    - Meter-read success
    - Recent-versus-historical changes

The final model uses:

    44 numerical features
    3 categorical features

Categorical features:

    hw_model
    site_type
    region

GATEWAY ID

gateway_id is deliberately excluded from the model.

It is an identifier rather than a direct measure of gateway health.
Excluding it reduces the risk of the model learning gateway-specific
identifiers instead of general patterns that can transfer to unseen
gateways.

------------------------------------------------------------
MODEL
------------------------------------------------------------

Model:

    Random Forest

The trained model is saved as:

    model_rf.pkl

The model generates gateway-level risk scores, which are then used
by predict.py to create the weekly top-15 rankings.

The trained .pkl model is included in the repository.

------------------------------------------------------------
VALIDATION
------------------------------------------------------------

FIVE-FOLD CROSS-VALIDATION

    Accuracy:           0.767
    Balanced accuracy:  0.767
    ROC-AUC:            0.848

GATEWAY-HELD-OUT VALIDATION

The model was also evaluated using gateway-held-out validation so
that gateways represented in validation were not used for training.

    Accuracy:           0.792
    Balanced accuracy:  0.792
    ROC-AUC:            0.850

TOP-15 VALIDATION

On the labelled engineer-review dataset:

    Average useful gateways: 10.8 / 15
    Precision@15:             0.720
    Recall:                   0.905

These top-15 metrics are validation results on the labelled
engineer-review dataset. They are not a claim about performance
across all gateways in the field.

OPERATIONAL COST LIMITATION

A complete eight-week operational euro-cost comparison is not
claimed.

The available historical field-visit records do not cover the
complete scored period through all eight prediction weeks.

Therefore, the available visit data is insufficient for a valid
complete eight-week operational cost backtest.

------------------------------------------------------------
WEEKLY PREDICTIONS
------------------------------------------------------------

The final submission generates predictions for the required
eight weeks:

    2026-02-02
    2026-02-09
    2026-02-16
    2026-02-23
    2026-03-02
    2026-03-09
    2026-03-16
    2026-03-23

For every week, the pipeline produces 15 ranked gateways with:

    week_start
    rank
    gateway_id
    score
    reason

The resulting file contains:

    8 weeks x 15 gateways = 120 rows

The generated predictions.csv passes the supplied submission
validator.

------------------------------------------------------------
REPRODUCIBILITY
------------------------------------------------------------

The complete workflow can be reproduced with:

    python run.py

The pipeline performs:

    Feature construction
          |
          v
    Model training
          |
          v
    Prediction generation
          |
          v
    Submission validation

Required Python packages are listed in:

    requirements.txt

------------------------------------------------------------
IMPORTANT FILES
------------------------------------------------------------

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

model_rf.pkl
    Trained Random Forest model.

DECISIONS.md
    Modelling decisions, alternatives considered, validation,
    limitations, and what additional data would improve the analysis.

AI-USAGE.md
    AI assistance and human verification record.

requirements.txt
    Python dependencies.

01-Challenge-Brief.pdf
    Challenge brief.

02-Data-Dictionary.pdf
    Data dictionary.

03-challenge-data.zip
    Challenge data archive.

baseline_3sigma.py
    Supplied baseline implementation.

23091A3245.pdf
    Participant resume.

------------------------------------------------------------
SUPPLIED BASELINE
------------------------------------------------------------

The original supplied 3-sigma baseline can be run independently:

    python baseline_3sigma.py --data data --out predictions.csv

The final submission, however, uses the ML pipeline:

    python run.py

------------------------------------------------------------
LIVE SESSION
------------------------------------------------------------

If invited to a live evaluation session, the model can be retrained
and the prediction pipeline rerun after receiving new data.

The system is intended as a prioritization system rather than a
guarantee that every selected gateway will fail.

------------------------------------------------------------
SCREEN RECORDING
------------------------------------------------------------

Project demonstration and complete pipeline walkthrough:

    https://drive.google.com/file/d/1um3am455ytVDwxO5cuej3eStXzqFIDp4/view?usp=drive_link

------------------------------------------------------------
DATA USAGE
------------------------------------------------------------

The challenge data is provided for this exercise only.

Do not pass the data on or publish it.

------------------------------------------------------------
DOCUMENTATION
------------------------------------------------------------

For detailed modelling decisions:

    DECISIONS.md

For AI usage and human verification:

    AI-USAGE.md

For the challenge requirements and data definitions:

    01-Challenge-Brief.pdf
    02-Data-Dictionary.pdf

------------------------------------------------------------
SUBMISSION OUTPUT
------------------------------------------------------------

The final submission artifact is:

    predictions.csv

It contains 120 rows covering the eight required prediction weeks,
with 15 ranked gateways per week, and passes the provided
submission validator.

============================================================
END OF README
============================================================