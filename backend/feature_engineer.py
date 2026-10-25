import pandas as pd

from features import build_features
from ml_config import FEATURES_FILE, RAW_FILE
from predict import load_bundle


# =====================================================
# LOAD DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard FEATURE ENGINEER")
print("======================================")

print("Input :", RAW_FILE)

df = pd.read_csv(RAW_FILE)

print("Rows  :", len(df))


# =====================================================
# VALIDATE REQUIRED COLUMNS
# =====================================================

required_columns = [
    "timestamp",
    "circuit_id",
    "presence",
    "temperature_C",
    "humidity_percent",
    "voltage_V",
    "current_A",
    "power_W",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


# =====================================================
# BASELINE FROM THE TRAINED MODEL
# =====================================================

# The model was trained against this baseline, so the
# pipeline must use the same one.

baseline = load_bundle()["baseline"]


# =====================================================
# FEATURES
# =====================================================

features = build_features(df, baseline)

features.to_csv(
    FEATURES_FILE,
    index=False
)


# =====================================================
# OUTPUT
# =====================================================

print()
print("Feature engineering complete.")

print("Output  :", FEATURES_FILE)
print("Rows    :", len(features))
print("Columns :", len(features.columns))
print("Sessions:", features["session_id"].nunique())

print()
print(features.tail())

print()
print("======================================")
