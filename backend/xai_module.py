import pickle
from pathlib import Path

import pandas as pd
import shap


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "c1_classified.csv"
MODEL_FILE = BASE_DIR / "data" / "xgboost_model.pkl"
ENCODER_FILE = BASE_DIR / "data" / "label_encoder.pkl"

OUTPUT_FILE = BASE_DIR / "data" / "c1_shap.csv"


# =====================================================
# FEATURES
# =====================================================

FEATURE_COLUMNS = [
    "power_W",
    "current_A",
    "voltage_V",
    "temperature_C",
    "humidity_percent",
    "presence_binary",
    "hour",
    "day_type",
    "rolling_mean_power",
    "rolling_std_power",
    "baseline_power",
    "deviation_from_baseline"
]


# =====================================================
# LOAD DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard SHAP EXPLAINER")
print("======================================")

print("Loading classified data...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))


# =====================================================
# SELECT ANOMALIES ONLY
# =====================================================

anomalies = df[
    df["anomaly"] == "ANOMALY"
].copy()

print("Anomalies:", len(anomalies))


if len(anomalies) == 0:

    print("No anomalies found.")
    print("Nothing to explain.")
    raise SystemExit


# =====================================================
# LOAD MODEL
# =====================================================

print()
print("Loading XGBoost model...")

with open(MODEL_FILE, "rb") as file:

    model = pickle.load(file)


with open(ENCODER_FILE, "rb") as file:

    label_encoder = pickle.load(file)


print("Model loaded.")


# =====================================================
# PREPARE INPUT
# =====================================================

X = anomalies[
    FEATURE_COLUMNS
].copy()

X = X.fillna(0)


# =====================================================
# SHAP EXPLAINER
# =====================================================

print()
print("Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("SHAP values calculated.")


# =====================================================
# HANDLE SHAP OUTPUT FORMAT
# =====================================================

def get_class_shap_values(
    shap_output,
    sample_index,
    class_index
):

    # Older SHAP format:
    # list of arrays
    if isinstance(shap_output, list):

        return shap_output[class_index][sample_index]

    # Newer SHAP format:
    # (samples, features, classes)
    if len(shap_output.shape) == 3:

        return shap_output[
            sample_index,
            :,
            class_index
        ]

    # Single-output fallback
    return shap_output[sample_index]


# =====================================================
# CREATE EXPLANATIONS
# =====================================================

results = []


for sample_index in range(len(X)):

    predicted_label = anomalies.iloc[
        sample_index
    ]["waste_type"]

    class_index = label_encoder.transform(
        [predicted_label]
    )[0]

    values = get_class_shap_values(
        shap_values,
        sample_index,
        class_index
    )


    # -------------------------------------------------
    # PAIR FEATURES WITH SHAP VALUES
    # -------------------------------------------------

    feature_importance = []

    for feature, value in zip(
        FEATURE_COLUMNS,
        values
    ):

        feature_importance.append(
            (
                feature,
                float(value)
            )
        )


    # -------------------------------------------------
    # SORT BY ABSOLUTE SHAP VALUE
    # -------------------------------------------------

    feature_importance.sort(
        key=lambda item: abs(item[1]),
        reverse=True
    )


    # -------------------------------------------------
    # TOP 3 FEATURES
    # -------------------------------------------------

    top_features = feature_importance[:3]


    row = {

        "timestamp":
            anomalies.iloc[sample_index]["timestamp"],

        "waste_type":
            predicted_label,

        "waste_confidence":
            anomalies.iloc[sample_index][
                "waste_confidence"
            ],

        "top1_feature":
            top_features[0][0],

        "top1_shap":
            round(top_features[0][1], 6),

        "top2_feature":
            top_features[1][0],

        "top2_shap":
            round(top_features[1][1], 6),

        "top3_feature":
            top_features[2][0],

        "top3_shap":
            round(top_features[2][1], 6)
    }

    results.append(row)


# =====================================================
# SAVE
# =====================================================

shap_df = pd.DataFrame(results)

shap_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# SHOW RESULTS
# =====================================================

print()
print("SHAP explanations:")

print(
    shap_df.to_string(index=False)
)

print()
print("Output:", OUTPUT_FILE)

print()
print("======================================")
print("SHAP EXPLANATION COMPLETE")
print("======================================")