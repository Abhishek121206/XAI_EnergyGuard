import pandas as pd
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "c1.csv"
OUTPUT_FILE = BASE_DIR / "data" / "c1_features.csv"


# =====================================================
# LOAD DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard FEATURE ENGINEER")
print("======================================")

print("Input :", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

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
# TIMESTAMP
# =====================================================

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="s"
)

df = df.sort_values("datetime")

df = df.reset_index(drop=True)


# =====================================================
# TIME FEATURES
# =====================================================

df["hour"] = df["datetime"].dt.hour

df["day_of_week"] = df["datetime"].dt.dayofweek

df["day_type"] = (
    df["day_of_week"]
    .apply(lambda x: 1 if x >= 5 else 0)
)


# =====================================================
# PRESENCE
# =====================================================

df["presence_binary"] = (
    df["presence"]
    .map({
        "PRESENT": 1,
        "ABSENT": 0
    })
)


# =====================================================
# POWER ROLLING FEATURES
# =====================================================

df["rolling_mean_power"] = (
    df["power_W"]
    .rolling(
        window=10,
        min_periods=1
    )
    .mean()
)

df["rolling_std_power"] = (
    df["power_W"]
    .rolling(
        window=10,
        min_periods=1
    )
    .std()
    .fillna(0)
)


# =====================================================
# BASELINE POWER
# =====================================================

df["baseline_power"] = (
    df.groupby(
        ["circuit_id", "hour"]
    )["power_W"]
    .transform("mean")
)


# =====================================================
# DEVIATION FROM BASELINE
# =====================================================

df["deviation_from_baseline"] = (
    df["power_W"]
    - df["baseline_power"]
)


# =====================================================
# FINAL FEATURE DATASET
# =====================================================

feature_columns = [

    "timestamp",

    "circuit_id",

    "power_W",

    "current_A",

    "voltage_V",

    "temperature_C",

    "humidity_percent",

    "presence_binary",

    "hour",

    "day_of_week",

    "day_type",

    "rolling_mean_power",

    "rolling_std_power",

    "baseline_power",

    "deviation_from_baseline"
]


features = df[feature_columns].copy()


# =====================================================
# REMOVE INVALID ROWS
# =====================================================

features = features.dropna(
    subset=[
        "power_W",
        "current_A",
        "temperature_C",
        "humidity_percent",
        "presence_binary"
    ]
)


# =====================================================
# SAVE
# =====================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# OUTPUT
# =====================================================

print()
print("Feature engineering complete.")

print("Output :", OUTPUT_FILE)
print("Rows   :", len(features))
print("Columns:", len(features.columns))

print()
print(features.head())

print()
print("======================================")