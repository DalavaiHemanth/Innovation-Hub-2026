# DECISIONS.md

## 1. Decision summary

The solution treats the task as a weekly gateway-risk ranking problem.

For each Monday in the eight-week scoring period, the model assigns every gateway a risk score and selects the top 15 gateways for the available field visits.

The final model is a Random Forest classifier using historical telemetry, meter-read behaviour, and a small set of gateway metadata. Gateway ID is deliberately excluded so that the model can generalise to gateways not seen during training.

The final submission contains 120 rows: 15 ranked gateways for each of the eight required weeks.

## 2. Why this framing

A gateway failure can silently stop readings from many meters. With only 15 field visits available each week, the practical objective is not to classify every gateway perfectly. The important decision is which 15 gateways should receive attention first.

The model is therefore evaluated as a ranking system, especially through Precision@15 and Recall@15.

## 3. Features used

The model uses information available before each prediction week.

### Telemetry

The feature engineering aggregates telemetry to gateway/day level and calculates 7-day, 14-day, and 28-day statistics.

Important groups include:

- disconnection counts
- offline duration
- system load
- uptime
- signal-quality indicators
- packet/CRC behaviour
- transmission success
- reboot behaviour

Signal-quality features include the proportion of observations in bad RSSI, RSRP/RSCP, and RSRQ/ECIO categories.

Recent-change features compare recent 7-day behaviour with the 28-day baseline for selected failure-related metrics.

### Meter-read behaviour

The model uses:

- mean meter-read success rate
- variation in meter-read success rate
- number of zero-read weeks

These provide an operational signal complementary to gateway telemetry.

### Metadata

Only three metadata fields are used:

- hardware model
- site type
- region

These provide context without allowing the model to memorise individual gateways.

## 4. What was deliberately omitted

### Gateway ID

Gateway ID is not used as a predictive feature.

This is important because the challenge requires testing on gateways never seen during training. Using an individual gateway identifier could allow the model to learn gateway-specific patterns instead of general failure signals.

After removing gateway ID, the model retained strong unseen-gateway performance.

### Firmware and antenna metadata

The final model does not use firmware version or antenna type.

An unseen-gateway comparison showed almost no improvement from adding the additional metadata:

- current metadata: ROC-AUC 0.850
- all metadata: ROC-AUC 0.852

The small difference was not considered sufficient to justify adding those variables.

### Field-visit outcomes and repair information

Historical field-visit outcomes, replaced parts, and engineer review comments are not used as prediction features.

These records are potentially selection-biased because field visits were already targeted by an operational process. Using them directly could make the model learn the historical intervention process rather than underlying gateway risk.

## 5. Model choice

A Random Forest was selected because it can combine heterogeneous telemetry features and categorical metadata while capturing nonlinear relationships and interactions.

The final model uses:

- 500 trees
- maximum depth of 4
- minimum leaf size of 4
- balanced class weighting
- fixed random seed

The constrained tree depth and minimum leaf size are intended to reduce overfitting on the small labelled review set.

## 6. Validation

The labelled engineer-review dataset contains 120 gateways:

- 60 Normal
- 60 Schlecht

### Five-fold cross-validation

The final model achieved:

- Accuracy: **0.767**
- Balanced accuracy: **0.767**
- ROC-AUC: **0.848**

### Unseen-gateway validation

The validation contains 120 unique gateways, so each validation prediction is for a gateway not used for fitting that fold.

The final model achieved:

- Accuracy: **0.792**
- Balanced accuracy: **0.792**
- ROC-AUC: **0.850**

Confusion matrix:

```text
[[45 15]
 [10 50]]