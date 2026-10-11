import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

TRAINING_FILE = BASE_DIR / "data" / "xgboost_training.csv"
ANOMALY_FILE = BASE_DIR / "data" / "c1_anomalies.csv"

MODEL_FILE = BASE_DIR / "data" / "xgboost_model.pkl"
ENCODER_FILE = BASE_DIR / "data" / "label_encoder.pkl"

OUTPUT_FILE = BASE_DIR / "data" / "c1_classified.csv"


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
# LOAD TRAINING DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard WASTE CLASSIFIER")
print("======================================")

print("Loading training data...")

training_df = pd.read_csv(TRAINING_FILE)

print("Training rows:", len(training_df))


# =====================================================
# PREPARE TRAINING DATA
# =====================================================

X = training_df[FEATURE_COLUMNS].copy()
y_text = training_df["label"].copy()


# =====================================================
# ENCODE LABELS
# =====================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

print()
print("Classes:")

for index, label in enumerate(label_encoder.classes_):

    print(index, "=", label)


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# XGBOOST MODEL
# =====================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    eval_metric="mlogloss",
    random_state=42
)


# =====================================================
# TRAIN
# =====================================================

print()
print("Training XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =====================================================
# VALIDATION
# =====================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("Validation accuracy:", round(accuracy, 4))

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


# =====================================================
# SAVE MODEL
# =====================================================

with open(MODEL_FILE, "wb") as file:

    pickle.dump(model, file)


with open(ENCODER_FILE, "wb") as file:

    pickle.dump(label_encoder, file)


print()
print("Model saved:", MODEL_FILE)
print("Encoder saved:", ENCODER_FILE)


# =====================================================
# CLASSIFY DETECTED ANOMALIES
# =====================================================

print()
print("Classifying detected anomalies...")

anomaly_df = pd.read_csv(ANOMALY_FILE)

anomalies = anomaly_df[
    anomaly_df["anomaly"] == "ANOMALY"
].copy()


if len(anomalies) == 0:

    print("No anomalies found.")

else:

    X_anomalies = anomalies[
        FEATURE_COLUMNS
    ].copy()

    X_anomalies = X_anomalies.fillna(0)

    anomaly_predictions = model.predict(
        X_anomalies
    )

    anomaly_probabilities = model.predict_proba(
        X_anomalies
    )

    anomaly_labels = (
        label_encoder
        .inverse_transform(anomaly_predictions)
    )

    anomaly_confidence = (
        anomaly_probabilities.max(axis=1)
    )

    anomalies["waste_type"] = anomaly_labels

    anomalies["waste_confidence"] = (
        anomaly_confidence.round(4)
    )


# =====================================================
# ADD CLASSIFICATION TO FULL DATASET
# =====================================================

anomaly_df["waste_type"] = "NORMAL"

anomaly_df["waste_confidence"] = 0.0


if len(anomalies) > 0:

    anomaly_df.loc[
        anomalies.index,
        "waste_type"
    ] = anomalies["waste_type"]

    anomaly_df.loc[
        anomalies.index,
        "waste_confidence"
    ] = anomalies["waste_confidence"]


# =====================================================
# SAVE CLASSIFIED DATA
# =====================================================

anomaly_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# SHOW RESULTS
# =====================================================

print()
print("Classification results:")

if len(anomalies) > 0:

    print(
        anomaly_df[
            [
                "timestamp",
                "power_W",
                "presence_binary",
                "anomaly",
                "waste_type",
                "waste_confidence"
            ]
        ].to_string(index=False)
    )

else:

    print("No anomalies to classify.")


print()
print("Output:", OUTPUT_FILE)

print()
print("======================================")
print("WASTE CLASSIFICATION COMPLETE")
print("======================================")