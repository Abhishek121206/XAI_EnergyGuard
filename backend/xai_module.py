import numpy as np
import pandas as pd
import shap

from ml_config import (
    CLASSIFIED_FILE,
    MAX_EXPLAIN_ROWS,
    SHAP_FILE
)
from predict import Predictor


TOP_N = 3


# =====================================================
# LOAD DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard SHAP EXPLAINER")
print("======================================")

print("Loading classified data...")

df = pd.read_csv(CLASSIFIED_FILE)

print("Total rows:", len(df))


# =====================================================
# SELECT WASTE EVENTS (most recent first)
# =====================================================

anomalies = (
    df[df["waste_type"] != "NORMAL"]
    .tail(MAX_EXPLAIN_ROWS)
    .copy()
)

print("Waste events to explain:", len(anomalies))


# =====================================================
# LOAD MODEL
# =====================================================

predictor = Predictor()

print("Model version:", predictor.version)


# =====================================================
# REUSE CACHED EXPLANATIONS
# =====================================================

# SHAP for a row only changes if the model or the
# predicted class changes.

cached = pd.DataFrame()

if SHAP_FILE.exists():

    previous = pd.read_csv(SHAP_FILE)

    if {"model_version", "reading_id"} <= set(previous.columns):

        cached = previous[
            previous["model_version"] == predictor.version
        ].merge(
            anomalies[["reading_id", "waste_type"]],
            on=["reading_id", "waste_type"]
        )

to_explain = anomalies[
    ~anomalies["reading_id"].isin(
        cached.get("reading_id", [])
    )
]

print("Cached:", len(cached), " New:", len(to_explain))


# =====================================================
# SHAP
# =====================================================

def get_class_shap_values(shap_output, class_indices):

    """Per-row SHAP values for each row's predicted class."""

    rows = np.arange(len(class_indices))

    # Older SHAP format: list of (samples, features)
    if isinstance(shap_output, list):

        stacked = np.stack(shap_output, axis=-1)

        return stacked[rows, :, class_indices]

    # Newer SHAP format: (samples, features, classes)
    if shap_output.ndim == 3:

        return shap_output[rows, :, class_indices]

    # Single-output fallback
    return shap_output


results = []

if len(to_explain) > 0:

    X = predictor.model_input(to_explain)

    print()
    print("Creating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(predictor.classifier)

    shap_values = explainer.shap_values(X)

    class_indices = np.array([
        list(predictor.classes).index(label)
        for label in to_explain["waste_type"]
    ])

    values = get_class_shap_values(
        shap_values,
        class_indices
    )

    print("SHAP values calculated.")

    for row_index, (_, row) in enumerate(
        to_explain.iterrows()
    ):

        # Sort features by absolute SHAP value

        order = np.argsort(
            -np.abs(values[row_index])
        )[:TOP_N]

        result = {
            "reading_id": row["reading_id"],
            "timestamp": row["timestamp"],
            "waste_type": row["waste_type"],
            "waste_confidence": row["waste_confidence"],
            "model_version": predictor.version
        }

        for rank, feature_index in enumerate(order, start=1):

            feature = predictor.features[feature_index]

            result[f"top{rank}_feature"] = feature
            result[f"top{rank}_value"] = round(
                float(row[feature]), 4
            )
            result[f"top{rank}_shap"] = round(
                float(values[row_index][feature_index]), 6
            )

        results.append(result)


# =====================================================
# SAVE
# =====================================================

shap_df = (
    pd.concat(
        [cached, pd.DataFrame(results)],
        ignore_index=True
    )
    .sort_values("timestamp", kind="stable")
)

shap_df.to_csv(
    SHAP_FILE,
    index=False
)


# =====================================================
# SHOW RESULTS
# =====================================================

print()
print("Most common top feature per class:")

if not shap_df.empty:

    print(
        shap_df
        .groupby("waste_type")["top1_feature"]
        .agg(lambda s: s.value_counts().index[0])
        .to_string()
    )

print()
print("Latest explanations:")

print(
    shap_df.tail(5).to_string(index=False)
)

print()
print("Output:", SHAP_FILE)

print()
print("======================================")
print("SHAP EXPLANATION COMPLETE")
print("======================================")
