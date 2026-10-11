import numpy as np
import pandas as pd
import dice_ml

from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_FILE = BASE_DIR / "data" / "c1_features.csv"
ANOMALY_FILE = BASE_DIR / "data" / "c1_anomalies.csv"

OUTPUT_FILE = BASE_DIR / "data" / "c1_counterfactuals.csv"


# =====================================================
# FEATURES
# =====================================================

MODEL_FEATURES = [
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
# LOAD FEATURE DATA
# =====================================================

print("======================================")
print("XAI-EnergyGuard DiCE")
print("======================================")

print("Loading feature data...")

features_df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(features_df))


# =====================================================
# PREPARE TRAINING DATA
# =====================================================

X_train = features_df[MODEL_FEATURES].copy()

X_train = X_train.fillna(0)


# =====================================================
# TRAIN ISOLATION FOREST
# =====================================================

print("Training Isolation Forest...")

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples=min(256, len(X_train)),
    random_state=42
)

isolation_forest.fit(X_train)

print("Isolation Forest ready.")


# =====================================================
# DiCE MODEL WRAPPER
# =====================================================

class IsolationForestWrapper(
    BaseEstimator,
    ClassifierMixin
):

    def __init__(self, model):

        self.model = model

    def _prepare_data(self, X):

        if isinstance(X, pd.DataFrame):

            data = X.copy()

        else:

            data = pd.DataFrame(
                X,
                columns=MODEL_FEATURES
            )

        data = data[MODEL_FEATURES].copy()

        data = data.fillna(0)

        # Keep current derived values consistent
        # when DiCE changes power_W.

        data["current_A"] = (
            data["power_W"] / 5.0
        )

        data["deviation_from_baseline"] = (
            data["power_W"]
            - data["baseline_power"]
        )

        return data

    def predict(self, X):

        data = self._prepare_data(X)

        isolation_predictions = (
            self.model.predict(data)
        )

        # Isolation Forest:
        #
        #  1  = normal
        # -1  = anomaly
        #
        # DiCE:
        #
        #  0  = normal
        #  1  = anomaly

        return np.where(
            isolation_predictions == -1,
            1,
            0
        )

    def predict_proba(self, X):

        data = self._prepare_data(X)

        scores = self.model.decision_function(data)

        # Convert the Isolation Forest decision score
        # into a probability-like anomaly value.

        anomaly_probability = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    np.clip(
                        scores * 10,
                        -50,
                        50
                    )
                )
            )
        )

        normal_probability = (
            1.0
            - anomaly_probability
        )

        return np.column_stack(
            [
                normal_probability,
                anomaly_probability
            ]
        )


# =====================================================
# CREATE WRAPPER
# =====================================================

wrapped_model = IsolationForestWrapper(
    isolation_forest
)


# =====================================================
# CREATE DiCE DATASET
# =====================================================

dice_data = features_df[
    MODEL_FEATURES
].copy()

dice_data = dice_data.fillna(0)

dice_data["anomaly_class"] = (
    wrapped_model.predict(dice_data)
)


# =====================================================
# DiCE DATA INTERFACE
# =====================================================

continuous_features = [
    "power_W",
    "current_A",
    "temperature_C",
    "humidity_percent",
    "rolling_mean_power",
    "rolling_std_power",
    "baseline_power",
    "deviation_from_baseline",
    "hour"
]

categorical_features = [
    "presence_binary",
    "day_type"
]


data_interface = dice_ml.Data(
    dataframe=dice_data,
    continuous_features=continuous_features,
    categorical_features=categorical_features,
    outcome_name="anomaly_class"
)


# =====================================================
# DiCE MODEL INTERFACE
# =====================================================

dice_model = dice_ml.Model(
    model=wrapped_model,
    backend="sklearn",
    model_type="classifier"
)


# =====================================================
# CREATE DiCE EXPLAINER
# =====================================================

explainer = dice_ml.Dice(
    data_interface,
    dice_model,
    method="random"
)


# =====================================================
# LOAD DETECTED ANOMALIES
# =====================================================

anomaly_df = pd.read_csv(
    ANOMALY_FILE
)

anomalies = anomaly_df[
    anomaly_df["anomaly"] == "ANOMALY"
].copy()

print()
print("Anomalies to explain:", len(anomalies))


if len(anomalies) == 0:

    print("No anomalies found.")
    raise SystemExit


# =====================================================
# GENERATE COUNTERFACTUALS
# =====================================================

results = []


for _, anomaly in anomalies.iterrows():

    timestamp = anomaly["timestamp"]

    original_power = float(
        anomaly["power_W"]
    )

    print()
    print(
        "Processing timestamp:",
        timestamp
    )

    # -------------------------------------------------
    # Build query
    # -------------------------------------------------

    query = pd.DataFrame(
        [
            {
                feature:
                    anomaly[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )

    query = query.fillna(0)

    # -------------------------------------------------
    # Generate counterfactuals
    # -------------------------------------------------

    try:

        explanation = (
            explainer.generate_counterfactuals(
                query,
                total_CFs=3,
                desired_class=0,
                features_to_vary=[
                    "power_W"
                ]
            )
        )

        if (
            explanation.cf_examples_list is None
            or
            len(explanation.cf_examples_list) == 0
        ):

            print("No counterfactual found.")

            results.append(
                {
                    "timestamp":
                        timestamp,

                    "original_power_W":
                        round(
                            original_power,
                            3
                        ),

                    "counterfactual_power_W":
                        None,

                    "power_reduction_W":
                        None,

                    "message":
                        "No counterfactual found"
                }
            )

            continue


        cf_df = (
            explanation
            .cf_examples_list[0]
            .final_cfs_df
        )

        if cf_df is None or len(cf_df) == 0:

            print("No counterfactual found.")

            results.append(
                {
                    "timestamp":
                        timestamp,

                    "original_power_W":
                        round(
                            original_power,
                            3
                        ),

                    "counterfactual_power_W":
                        None,

                    "power_reduction_W":
                        None,

                    "message":
                        "No counterfactual found"
                }
            )

            continue


        # -------------------------------------------------
        # ONLY ACCEPT POWER REDUCTIONS
        # -------------------------------------------------

        valid_cf = cf_df[
            cf_df["power_W"] < original_power
        ].copy()


        if len(valid_cf) == 0:

            print(
                "No valid power-reducing "
                "counterfactual found."
            )

            results.append(
                {
                    "timestamp":
                        timestamp,

                    "original_power_W":
                        round(
                            original_power,
                            3
                        ),

                    "counterfactual_power_W":
                        None,

                    "power_reduction_W":
                        None,

                    "message":
                        "No valid power-reducing "
                        "counterfactual found"
                }
            )

            continue


        # -------------------------------------------------
        # FIND CLOSEST POWER REDUCTION
        # -------------------------------------------------

        valid_cf["power_distance"] = (
            original_power
            - valid_cf["power_W"]
        )

        best_cf = (
            valid_cf
            .sort_values("power_distance")
            .iloc[0]
        )


        counterfactual_power = float(
            best_cf["power_W"]
        )


        power_reduction = (
            original_power
            - counterfactual_power
        )


        # -------------------------------------------------
        # CREATE MESSAGE
        # -------------------------------------------------

        message = (
            "Reducing power from "
            f"{original_power:.3f} W to "
            f"approximately "
            f"{counterfactual_power:.3f} W "
            "would move this reading toward "
            "the normal region of the anomaly detector."
        )


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        results.append(
            {
                "timestamp":
                    timestamp,

                "original_power_W":
                    round(
                        original_power,
                        3
                    ),

                "counterfactual_power_W":
                    round(
                        counterfactual_power,
                        3
                    ),

                "power_reduction_W":
                    round(
                        power_reduction,
                        3
                    ),

                "message":
                    message
            }
        )


        print(
            "Original power:",
            round(
                original_power,
                3
            ),
            "W"
        )

        print(
            "Counterfactual:",
            round(
                counterfactual_power,
                3
            ),
            "W"
        )

        print(
            "Power reduction:",
            round(
                power_reduction,
                3
            ),
            "W"
        )


    except Exception as error:

        print(
            "DiCE error:",
            error
        )

        results.append(
            {
                "timestamp":
                    timestamp,

                "original_power_W":
                    round(
                        original_power,
                        3
                    ),

                "counterfactual_power_W":
                    None,

                "power_reduction_W":
                    None,

                "message":
                    "DiCE error: "
                    + str(error)
            }
        )


# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print()
print("======================================")
print("COUNTERFACTUAL RESULTS")
print("======================================")

if len(results_df) > 0:

    print(
        results_df.to_string(
            index=False
        )
    )

else:

    print("No results generated.")


print()
print("Output:", OUTPUT_FILE)

print()
print("======================================")
print("DiCE COMPLETE")
print("======================================")