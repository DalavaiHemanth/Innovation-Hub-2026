# Decisions

## 1. Part 2 area: Machine Learning

I selected **Machine Learning (ML)** as the main Part 2 area.

### Why

The task is a weekly gateway triage problem where the field team can visit only 15 gateways. A supervised model can learn patterns associated with gateways that engineers have classified as `Schlecht`, then rank all gateways by predicted risk.

ML also provides a natural way to combine multiple telemetry signals instead of relying on a single anomaly threshold.

### Alternatives considered

- **Data Engineering:** useful for building a reliable feature pipeline, but by itself does not directly solve the ranking/prediction problem.
- **Decision Engineering:** useful for optimizing the final selection under the 15-visit constraint, but it depends on having a good risk signal first.

I therefore chose ML as the primary approach, while using data-engineering ideas in the feature pipeline and a ranking decision at prediction time.

---

## 2. Use multi-window telemetry features instead of only raw recent values

I used gateway-level features calculated over **7, 14 and 28 day windows** before each prediction week.

The model uses information such as:

- disconnection counts
- offline duration
- reboot counts
- average load
- uptime
- signal quality
- packet/CRC related metrics
- transmission metrics
- active days
- recent-vs-history changes
- meter-read success statistics

### Why

Gateway failures can appear as persistent problems or as a change from a gateway's normal behaviour. Multiple time windows capture both effects.

The 28-day window provides a longer baseline, while 7-day and 14-day windows capture more recent changes.

### Alternative considered

A simpler approach was to rank gateways using only disconnections, or only the same three signals used by the supplied 3-sigma baseline.

I tested simpler ranking approaches. The Random Forest produced better top-15 validation results than the simple disconnection-based comparison.

---

## 3. Use Random Forest instead of Logistic Regression

I tested both Logistic Regression and Random Forest models.

The final model is a Random Forest with:

- 500 trees
- maximum depth = 4
- minimum leaf size = 4
- balanced class weighting
- fixed random seed

The final training pipeline uses 44 numeric features and 3 categorical features:

- `hw_model`
- `site_type`
- `region`

### Why

The Random Forest performed better during validation and can model non-linear relationships between telemetry behaviour and gateway condition.

Five-fold cross-validation gave:

- Accuracy: **0.767**
- Balanced accuracy: **0.767**
- ROC-AUC: **0.848**

### Alternative considered

Logistic Regression was tested first because it is simple and interpretable. Its validation performance was lower, so Random Forest was selected.

---

## 4. Do not use gateway ID as a predictive feature

I explicitly excluded `gateway_id` from the final model features.

### Why

Gateway IDs are identifiers rather than measurements of gateway health. Including them could allow the model to memorize gateway-specific patterns instead of learning general failure behaviour.

This is especially important because the challenge requires testing on gateways that were not seen during training.

### Alternative considered

Including gateway ID was initially considered because it is available in the data and can sometimes capture persistent device-specific effects.

I rejected this approach because it would weaken generalization to unseen gateways and make the model harder to justify operationally.

The final model therefore uses operational telemetry and selected metadata instead of the gateway identifier.

---

## 5. Generate predictions strictly from information available before each week

For every scored Monday, the prediction pipeline uses historical information strictly before that Monday.

The model then produces a risk score for all available gateways, ranks them, and selects the top 15.

### Why

The field team has a hard capacity limit of 15 visits per week. The output therefore needs to be a ranked triage list rather than a binary prediction for every gateway.

Using only information available before the prediction week prevents future information from entering the prediction process.

### Alternative considered

Using information from the full observation period could produce stronger-looking retrospective results, but it would introduce future information into earlier predictions.

I rejected that approach because it would not represent how the system would operate in production.

---

# Validation and results

The final Random Forest was evaluated using cross-validation and a gateway-held-out validation procedure.

### Cross-validation

- Accuracy: **0.767**
- Balanced accuracy: **0.767**
- ROC-AUC: **0.848**

### Gateway-held-out validation

- Accuracy: **0.792**
- Balanced accuracy: **0.792**
- ROC-AUC: **0.850**
- Confusion matrix: `[[45, 15], [10, 50]]`

This validation keeps gateways separate between training and validation, providing a more realistic test of generalization to gateways not seen during training.

### Top-15 validation

Across the five validation folds:

- Average useful gateways selected: **10.8 / 15**
- Precision@15: **0.720**
- Recall: **0.905**

The simple comparison baselines achieved approximately:

- **9.6 / 15** useful gateways
- Precision@15: **0.640**
- Recall: **0.799**

These top-15 results are validation results on the labelled engineer-review dataset; they are not claimed as an eight-week field-cost measurement.

---

# Prediction procedure

The final prediction pipeline:

1. Loads telemetry, meter-read data and gateway metadata.
2. Builds features using historical data before each prediction week.
3. Loads the trained Random Forest.
4. Scores all gateways.
5. Ranks gateways by model score.
6. Selects the top 15 gateways.
7. Generates a reason based on the strongest recent operational indicators.
8. Produces the required `predictions.csv`.

The final submission contains:

- **8 weeks**
- **15 gateways per week**
- **120 total rows**

The supplied validator reports:

`predictions.csv: OK`

---

# Limitations

The labelled engineer-review dataset contains only 120 reviewed gateways, so the supervised model is trained on a relatively small sample.

Telemetry history is also incomplete for some reviewed gateways.

The engineer-review labels may contain selection bias because the reviewed gateways were not necessarily a random sample of all gateways.

The available field-visit records do not provide complete observed outcomes for all eight scored weeks. Therefore, a complete eight-week operational cost comparison cannot be claimed from the available historical visit data.

The model should therefore be treated as a prioritization system rather than as a guarantee that every selected gateway will fail.

---

# What two more weeks of data would improve

Two additional weeks of labelled operational data would help with:

1.**Recent failure patterns**
More recent labels would reveal whether the learned relationships still hold under current network conditions.

2.**Calibration and drift checking**
New observations could be used to check whether predicted risk remains aligned with observed gateway problems and whether network changes have altered the telemetry patterns.

The additional data could then be used for retraining and a fresh gateway-held-out validation.