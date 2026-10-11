import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "c1_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "xgboost_training.csv"


# =====================================================
# SETTINGS
# =====================================================

np.random.seed(42)

SAMPLES_PER_CLASS = 500


# =====================================================
# LOAD EXISTING DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)


# =====================================================
# BASE VALUES FROM YOUR REAL TEST DATA
# =====================================================

power_mean = df["power_W"].mean()
power_std = df["power_W"].std()

current_mean = df["current_A"].mean()
current_std = df["current_A"].std()

temperature_mean = df["temperature_C"].mean()
temperature_std = df["temperature_C"].std()

humidity_mean = df["humidity_percent"].mean()
humidity_std = df["humidity_percent"].std()

baseline_mean = df["baseline_power"].mean()


# Avoid zero standard deviation
if power_std == 0:
    power_std = 0.1

if current_std == 0:
    current_std = 0.01

if temperature_std == 0:
    temperature_std = 1

if humidity_std == 0:
    humidity_std = 1


# =====================================================
# GENERATE TRAINING SAMPLES
# =====================================================

rows = []


# =====================================================
# IDLE EQUIPMENT
# =====================================================

for _ in range(SAMPLES_PER_CLASS):

    baseline = max(
        0.1,
        np.random.normal(
            baseline_mean,
            power_std
        )
    )

    power = max(
        0.1,
        np.random.normal(
            baseline * 1.5,
            power_std
        )
    )

    current = power / 5.0

    rows.append({

        "power_W": power,
        "current_A": current,
        "voltage_V": 5.0,

        "temperature_C":
            np.random.normal(
                temperature_mean,
                temperature_std
            ),

        "humidity_percent":
            np.random.normal(
                humidity_mean,
                humidity_std
            ),

        "presence_binary": 0,

        "hour":
            np.random.randint(0, 24),

        "day_type":
            np.random.randint(0, 2),

        "rolling_mean_power":
            baseline,

        "rolling_std_power":
            abs(np.random.normal(0.2, 0.1)),

        "baseline_power":
            baseline,

        "deviation_from_baseline":
            power - baseline,

        "label": "IE"
    })


# =====================================================
# OVERCONSUMPTION
# =====================================================

for _ in range(SAMPLES_PER_CLASS):

    baseline = max(
        0.1,
        np.random.normal(
            baseline_mean,
            power_std
        )
    )

    power = max(
        baseline * 2.2,
        np.random.normal(
            baseline * 2.5,
            power_std
        )
    )

    current = power / 5.0

    rows.append({

        "power_W": power,
        "current_A": current,
        "voltage_V": 5.0,

        "temperature_C":
            np.random.normal(
                temperature_mean,
                temperature_std
            ),

        "humidity_percent":
            np.random.normal(
                humidity_mean,
                humidity_std
            ),

        "presence_binary": 1,

        "hour":
            np.random.randint(0, 24),

        "day_type":
            np.random.randint(0, 2),

        "rolling_mean_power":
            baseline,

        "rolling_std_power":
            abs(np.random.normal(0.5, 0.2)),

        "baseline_power":
            baseline,

        "deviation_from_baseline":
            power - baseline,

        "label": "OC"
    })


# =====================================================
# UNUSUAL PATTERN
# =====================================================

for _ in range(SAMPLES_PER_CLASS):

    baseline = max(
        0.1,
        np.random.normal(
            baseline_mean,
            power_std
        )
    )

    power = max(
        0.1,
        np.random.normal(
            baseline * 1.3,
            power_std * 2
        )
    )

    current = power / 5.0

    rows.append({

        "power_W": power,
        "current_A": current,
        "voltage_V": 5.0,

        "temperature_C":
            np.random.normal(
                temperature_mean,
                temperature_std * 3
            ),

        "humidity_percent":
            np.random.normal(
                humidity_mean,
                humidity_std * 3
            ),

        "presence_binary":
            np.random.randint(0, 2),

        "hour":
            np.random.randint(0, 24),

        "day_type":
            np.random.randint(0, 2),

        "rolling_mean_power":
            baseline,

        "rolling_std_power":
            abs(np.random.normal(2.0, 0.7)),

        "baseline_power":
            baseline,

        "deviation_from_baseline":
            power - baseline,

        "label": "UP"
    })


# =====================================================
# CREATE DATAFRAME
# =====================================================

training_df = pd.DataFrame(rows)


# =====================================================
# CLEAN DATA
# =====================================================

training_df = training_df.replace(
    [np.inf, -np.inf],
    np.nan
)

training_df = training_df.dropna()


# =====================================================
# SHUFFLE
# =====================================================

training_df = training_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =====================================================
# SAVE
# =====================================================

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# OUTPUT
# =====================================================

print("======================================")
print("XGBOOST TRAINING DATA")
print("======================================")

print("Output:", OUTPUT_FILE)

print()
print("Total rows:", len(training_df))

print()
print("Class distribution:")
print(
    training_df["label"].value_counts()
)

print()
print(training_df.head())

print()
print("Training data created successfully.")
print("======================================")