import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from features import (
    apply_baseline,
    build_features,
    compute_baseline,
    make_labels,
    rule_class
)
from ml_config import (
    CLASSES,
    METRICS_FILE,
    MODEL_BUNDLE_FILE,
    MODEL_DIR,
    MODEL_FEATURES,
    RAW_FILE,
    ROLLING_WINDOW
)


# =====================================================
# SETTINGS
# =====================================================

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Rows dropped between splits so rolling windows don't leak
SPLIT_GAP = ROLLING_WINDOW

ISOLATION_CONTAMINATION = 0.02

RANDOM_STATE = 42

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASSES)
}


# =====================================================
# HELPERS
# =====================================================

def new_classifier(n_estimators=600, early_stopping=True):

    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="mlogloss",
        early_stopping_rounds=40 if early_stopping else None,
        random_state=RANDOM_STATE
    )


def new_isolation_forest():

    return IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_STATE
    )


def evaluate(y_true, y_pred):

    """Metrics for label arrays of class names."""

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0
            ),
            4
        ),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"])
            }
            for label in CLASSES
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=CLASSES
            ).tolist()
        }
    }


def time_range(frame):

    return {
        "rows": int(len(frame)),
        "start": pd.to_datetime(
            frame["timestamp"].min(), unit="s"
        ).isoformat(),
        "end": pd.to_datetime(
            frame["timestamp"].max(), unit="s"
        ).isoformat()
    }


def print_metrics(name, metrics):

    print()
    print(name)
    print(
        "  accuracy:", metrics["accuracy"],
        " macro F1:", metrics["macro_f1"]
    )

    for label, values in metrics["per_class"].items():
        print(
            f"  {label:<7}"
            f" precision {values['precision']:.3f}"
            f"  recall {values['recall']:.3f}"
            f"  f1 {values['f1']:.3f}"
            f"  n={values['support']}"
        )


# =====================================================
# LOAD + FEATURES + LABELS
# =====================================================

print("======================================")
print("XAI-EnergyGuard MODEL TRAINING")
print("======================================")

print("Raw data:", RAW_FILE)

raw_df = pd.read_csv(RAW_FILE)

features_df = build_features(raw_df)
features_df["label"] = make_labels(features_df)

print("Rows    :", len(features_df))
print()
print("Label counts:")
print(features_df["label"].value_counts().to_string())


# =====================================================
# CHRONOLOGICAL SPLIT
# =====================================================

n_rows = len(features_df)

train_end = int(n_rows * TRAIN_FRACTION)
validation_end = int(
    n_rows * (TRAIN_FRACTION + VALIDATION_FRACTION)
)

train_df = features_df.iloc[:train_end].copy()
validation_df = features_df.iloc[
    train_end + SPLIT_GAP:validation_end
].copy()
test_df = features_df.iloc[
    validation_end + SPLIT_GAP:
].copy()

# Baseline uses training rows only, then is applied everywhere

train_baseline = compute_baseline(train_df)

for frame in (train_df, validation_df, test_df):
    apply_baseline(frame, train_baseline)

for name, frame in (
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df)
):

    counts = frame["label"].value_counts().to_dict()

    print()
    print(f"{name:<10}", time_range(frame))
    print(" " * 10, counts)

missing_train = [
    label
    for label in CLASSES
    if label not in set(train_df["label"])
]

if missing_train:

    raise SystemExit(
        "Training split has no rows for: "
        + ", ".join(missing_train)
        + ". Collect more data covering these cases."
    )


# =====================================================
# TRAIN + EVALUATE (held-out test)
# =====================================================

def xy(frame):

    return (
        frame[MODEL_FEATURES].astype(float),
        frame["label"].map(CLASS_TO_INDEX).to_numpy()
    )


X_train, y_train = xy(train_df)
X_validation, y_validation = xy(validation_df)
X_test, y_test = xy(test_df)

print()
print("Training XGBoost (early stopping on validation)...")

model = new_classifier()

model.fit(
    X_train,
    y_train,
    sample_weight=compute_sample_weight(
        "balanced", y_train
    ),
    eval_set=[(X_validation, y_validation)],
    sample_weight_eval_set=[
        compute_sample_weight(
            "balanced", y_validation
        )
    ],
    verbose=False
)

best_rounds = int(model.best_iteration) + 1

print("Best number of trees:", best_rounds)

test_labels = test_df["label"].to_numpy()

xgb_test_pred = np.array(CLASSES)[
    model.predict(X_test)
]

xgb_metrics = evaluate(test_labels, xgb_test_pred)

# -------------------------------------------------
# Baseline 1: the firmware rule (cannot detect UP)
# -------------------------------------------------

rule_metrics = evaluate(
    test_labels,
    rule_class(test_df).to_numpy()
)

# -------------------------------------------------
# Baseline 2: logistic regression
# -------------------------------------------------

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000
    )
)

logistic.fit(X_train, y_train)

logistic_metrics = evaluate(
    test_labels,
    np.array(CLASSES)[logistic.predict(X_test)]
)

# -------------------------------------------------
# Isolation Forest: outlier rate per class on test
# -------------------------------------------------

test_isolation = new_isolation_forest()

test_isolation.fit(
    X_train[train_df["label"].to_numpy() == "NORMAL"]
)

test_outliers = test_isolation.predict(X_test) == -1

outlier_rate = {
    label: round(
        float(test_outliers[test_labels == label].mean()),
        4
    ) if (test_labels == label).any() else None
    for label in CLASSES
}

print_metrics("XGBoost (test)", xgb_metrics)
print_metrics("Firmware rule baseline (test)", rule_metrics)
print_metrics("Logistic regression (test)", logistic_metrics)

print()
print("Isolation Forest outlier rate on test, by class:")
print(" ", outlier_rate)


# =====================================================
# FINAL MODEL (all data)
# =====================================================

print()
print("Refitting final model on all data...")

final_baseline = compute_baseline(features_df)

apply_baseline(features_df, final_baseline)

X_all, y_all = xy(features_df)

final_model = new_classifier(
    n_estimators=best_rounds,
    early_stopping=False
)

final_model.fit(
    X_all,
    y_all,
    sample_weight=compute_sample_weight(
        "balanced", y_all
    )
)

final_isolation = new_isolation_forest()

final_isolation.fit(
    X_all[features_df["label"].to_numpy() == "NORMAL"]
)

feature_importance = dict(
    sorted(
        zip(
            MODEL_FEATURES,
            (
                round(float(value), 4)
                for value in final_model.feature_importances_
            )
        ),
        key=lambda item: item[1],
        reverse=True
    )
)


# =====================================================
# SAVE
# =====================================================

MODEL_DIR.mkdir(exist_ok=True)

trained_at = datetime.now(timezone.utc).isoformat(
    timespec="seconds"
)

bundle = {
    "version": trained_at,
    "classes": CLASSES,
    "features": MODEL_FEATURES,
    "baseline": final_baseline,
    "classifier": final_model,
    "isolation_forest": final_isolation
}

joblib.dump(bundle, MODEL_BUNDLE_FILE)

metrics = {
    "trained_at": trained_at,
    "data_file": RAW_FILE.name,
    "rows": int(n_rows),
    "features": MODEL_FEATURES,
    "label_counts": {
        label: int(count)
        for label, count in (
            features_df["label"]
            .value_counts()
            .items()
        )
    },
    "splits": {
        "method": "chronological 70/15/15 with "
                  f"{SPLIT_GAP}-row gaps",
        "train": time_range(train_df),
        "validation": time_range(validation_df),
        "test": time_range(test_df)
    },
    "best_rounds": best_rounds,
    "test": {
        "xgboost": xgb_metrics,
        "firmware_rule_baseline": rule_metrics,
        "logistic_regression": logistic_metrics
    },
    "isolation_forest_outlier_rate_by_class": outlier_rate,
    "feature_importance": feature_importance,
    "label_note": (
        "IE/OC/NORMAL labels come from the ESP32 firmware rules "
        "(presence + 1.5 W limit); UP marks occupied readings "
        "with a sudden power change. High IE/OC scores therefore "
        "show the model reproduces those rules, not independent "
        "ground truth."
    )
}

METRICS_FILE.write_text(
    json.dumps(metrics, indent=2)
)

print()
print("Model  :", MODEL_BUNDLE_FILE)
print("Metrics:", METRICS_FILE)

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")
