import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "c1_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "c1_anomalies.csv"


# =====================================================
# LOAD DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard ANOMALY DETECTOR")
print("======================================")

print("Input :", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("Rows  :", len(df))


# =====================================================
# FEATURES USED BY ISOLATION FOREST
# =====================================================

feature_columns = [
    "power_W",
    "current_A",
    "temperature_C",
    "humidity_percent",
    "presence_binary",
    "rolling_mean_power",
    "rolling_std_power",
    "baseline_power",
    "deviation_from_baseline",
    "hour",
    "day_type"
]


# =====================================================
# CHECK FEATURES
# =====================================================

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns: "
        + ", ".join(missing_features)
    )


# =====================================================
# PREPARE ML DATA
# =====================================================

X = df[feature_columns].copy()

X = X.fillna(0)


# =====================================================
# ISOLATION FOREST
# =====================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples=256,
    random_state=42
)


# =====================================================
# TRAIN + PREDICT
# =====================================================

print()
print("Training Isolation Forest...")

predictions = model.fit_predict(X)

scores = model.decision_function(X)


# =====================================================
# ADD RESULTS
# =====================================================

df["anomaly_flag"] = predictions

df["anomaly_score"] = scores

df["anomaly"] = df["anomaly_flag"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})


# =====================================================
# SAVE
# =====================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

normal_count = (
    df["anomaly_flag"] == 1
).sum()

anomaly_count = (
    df["anomaly_flag"] == -1
).sum()


print()
print("Detection complete.")

print("Normal readings :", normal_count)
print("Anomalies        :", anomaly_count)

print()
print("Output :", OUTPUT_FILE)

print()
print(
    df[
        [
            "timestamp",
            "power_W",
            "presence_binary",
            "anomaly_score",
            "anomaly"
        ]
    ].to_string(index=False)
)

print()
print("======================================")