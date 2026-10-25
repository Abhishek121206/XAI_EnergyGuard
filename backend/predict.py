import json
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd

from features import build_features
from ml_config import (
    CLASSIFIED_FILE,
    FEATURES_FILE,
    MODEL_BUNDLE_FILE,
    ROLLING_WINDOW,
    SESSION_GAP_SECONDS
)


# =====================================================
# MODEL LOADING
# =====================================================

def load_bundle(bundle_path=MODEL_BUNDLE_FILE):

    if not bundle_path.exists():

        raise FileNotFoundError(
            f"No trained model at {bundle_path}. "
            "Run: python train.py"
        )

    return joblib.load(bundle_path)


# =====================================================
# PREDICTOR
# =====================================================

class Predictor:

    """Classifies readings as NORMAL / IE / OC / UP.

    predict_frame()   - batch, on engineered features
    predict_reading() - live, one raw MQTT reading at a time
    """

    def __init__(self, bundle_path=MODEL_BUNDLE_FILE):

        bundle = load_bundle(bundle_path)

        self.version = bundle["version"]
        self.classes = np.array(bundle["classes"])
        self.features = bundle["features"]
        self.baseline = bundle["baseline"]
        self.classifier = bundle["classifier"]
        self.isolation_forest = bundle["isolation_forest"]

        # Enough raw history to fill the rolling window
        self._buffer = deque(maxlen=ROLLING_WINDOW + 1)

    # -------------------------------------------------

    def model_input(self, features_df):

        return (
            features_df[self.features]
            .astype(float)
            .fillna(0.0)
        )

    # -------------------------------------------------

    def predict_frame(self, features_df):

        result = features_df.copy()

        if result.empty:

            for column in (
                "waste_type", "waste_confidence",
                "anomaly_score", "is_outlier",
                "anomaly_flag", "anomaly"
            ):
                result[column] = []

            return result

        X = self.model_input(result)

        probabilities = self.classifier.predict_proba(X)

        result["waste_type"] = self.classes[
            probabilities.argmax(axis=1)
        ]

        result["waste_confidence"] = (
            probabilities.max(axis=1).round(4)
        )

        for index, label in enumerate(self.classes):

            result[f"prob_{label}"] = (
                probabilities[:, index].round(4)
            )

        # Isolation Forest: secondary "how unusual" signal.
        # Lower score = more unusual.

        result["anomaly_score"] = (
            self.isolation_forest
            .decision_function(X)
            .round(4)
        )

        result["is_outlier"] = (
            self.isolation_forest.predict(X) == -1
        ).astype(int)

        is_waste = result["waste_type"] != "NORMAL"

        result["anomaly_flag"] = np.where(is_waste, -1, 1)

        result["anomaly"] = np.where(
            is_waste,
            "ANOMALY",
            "NORMAL"
        )

        return result

    # -------------------------------------------------

    def predict_raw(self, raw_df):

        return self.predict_frame(
            build_features(raw_df, self.baseline)
        )

    # -------------------------------------------------

    def predict_reading(self, reading):

        """Classify one raw reading (dict from the ESP32).

        Keeps a short history so rolling features match the
        batch pipeline. Returns a JSON-safe dict.
        """

        reading = dict(reading)

        reading.setdefault("timestamp", int(time.time()))

        if (
            self._buffer
            and reading["timestamp"]
            - self._buffer[-1]["timestamp"]
            > SESSION_GAP_SECONDS
        ):
            self._buffer.clear()

        self._buffer.append(reading)

        predictions = self.predict_raw(
            pd.DataFrame(list(self._buffer))
        )

        if predictions.empty:
            return None

        latest = predictions.iloc[-1]

        result = json.loads(latest.to_json())

        result["model_version"] = self.version

        return result


# =====================================================
# PIPELINE STAGE
# =====================================================

def main():

    print("======================================")
    print("XAI-EnergyGuard PREDICTION")
    print("======================================")

    predictor = Predictor()

    print("Model version:", predictor.version)

    features_df = pd.read_csv(FEATURES_FILE)

    print("Rows:", len(features_df))

    classified = predictor.predict_frame(features_df)

    classified.to_csv(CLASSIFIED_FILE, index=False)

    print()
    print("Predicted classes:")
    print(
        classified["waste_type"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Isolation Forest outliers:",
        int(classified["is_outlier"].sum())
    )

    print()
    print("Output:", CLASSIFIED_FILE)

    print()
    print("======================================")
    print("PREDICTION COMPLETE")
    print("======================================")


if __name__ == "__main__":

    main()
