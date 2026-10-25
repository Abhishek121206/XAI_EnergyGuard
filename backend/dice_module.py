import warnings

import numpy as np
import pandas as pd

from ml_config import (
    CLASSIFIED_FILE,
    COUNTERFACTUAL_FILE,
    LOAD_VOLTAGE,
    MAX_DICE_ROWS,
    MAX_EXPLAIN_ROWS,
    MODEL_FEATURES,
    OVERCONSUMPTION_LIMIT_W,
    ROLLING_WINDOW,
    SPIKE_MIN_W,
    SPIKE_STD_MULTIPLIER
)
from predict import Predictor

try:
    import dice_ml
except ImportError:
    dice_ml = None


# =====================================================
# SETTINGS
# =====================================================

# Candidate power values tried per reading by the model search
GRID_POINTS = 150

# Recent history (previous window) and presence are held
# fixed; only the power of the current reading is varied.

CONTEXT_COLUMNS = [
    "power_W",
    "presence_binary",
    "prev_mean_power",
    "prev_rolling_std_power",
    "rolling_std_power",
    "baseline_power"
]


print("======================================")
print("XAI-EnergyGuard COUNTERFACTUALS (DiCE)")
print("======================================")


# =====================================================
# CONTEXT <-> MODEL FEATURES
# =====================================================

def to_context(features_df):

    context = pd.DataFrame(index=features_df.index)

    context["power_W"] = features_df["power_W"]
    context["presence_binary"] = features_df["presence_binary"]
    context["prev_mean_power"] = (
        features_df["power_W"] - features_df["power_jump"]
    )
    context["prev_rolling_std_power"] = (
        features_df["prev_rolling_std_power"]
    )
    context["rolling_std_power"] = features_df["rolling_std_power"]
    context["baseline_power"] = features_df["baseline_power"]

    return context[CONTEXT_COLUMNS].astype(float)


def context_to_features(context):

    """Recompute every power-derived feature for a new power_W."""

    power = context["power_W"]
    previous_mean = context["prev_mean_power"]

    features = pd.DataFrame(index=context.index)

    features["power_W"] = power
    features["current_A"] = power / LOAD_VOLTAGE
    features["presence_binary"] = context["presence_binary"]
    features["rolling_mean_power"] = (
        previous_mean
        + (power - previous_mean) / ROLLING_WINDOW
    )
    features["rolling_std_power"] = context["rolling_std_power"]
    features["prev_rolling_std_power"] = (
        context["prev_rolling_std_power"]
    )
    features["power_jump"] = power - previous_mean
    features["power_jump_score"] = (
        features["power_jump"].abs()
        / context["prev_rolling_std_power"].clip(
            lower=SPIKE_MIN_W / SPIKE_STD_MULTIPLIER
        )
    )
    features["deviation_from_baseline"] = (
        power - context["baseline_power"]
    )

    return features[MODEL_FEATURES]


# =====================================================
# "IS THIS READING NORMAL?" MODEL
# =====================================================

class NormalityModel:

    """Binary view of the XGBoost classifier for DiCE.

    1 = NORMAL, 0 = any waste class.
    """

    def __init__(self, predictor):

        self.classifier = predictor.classifier
        self.normal_index = list(predictor.classes).index("NORMAL")

    def class_probabilities(self, context):

        if not isinstance(context, pd.DataFrame):
            context = pd.DataFrame(context, columns=CONTEXT_COLUMNS)

        return self.classifier.predict_proba(
            context_to_features(context[CONTEXT_COLUMNS])
        )

    def predict(self, context):

        return (
            self.class_probabilities(context).argmax(axis=1)
            == self.normal_index
        ).astype(int)

    def predict_proba(self, context):

        normal = self.class_probabilities(context)[
            :, self.normal_index
        ]

        return np.column_stack([1.0 - normal, normal])


# =====================================================
# MODEL-BASED SEARCH
# =====================================================

def power_upper_bound(context):

    return (
        context[["power_W", "prev_mean_power", "baseline_power"]]
        .max(axis=1)
        .clip(lower=2.0)
        * 1.5
    )


def search_counterfactual_power(model, context):

    """Closest power_W (to the original) the model calls NORMAL.

    Returns an array with NaN where no power value works.
    """

    n_rows = len(context)

    candidates = (
        np.linspace(0.0, 1.0, GRID_POINTS)[None, :]
        * power_upper_bound(context).to_numpy()[:, None]
    )

    expanded = context.loc[
        context.index.repeat(GRID_POINTS)
    ].reset_index(drop=True)

    expanded["power_W"] = candidates.ravel()

    is_normal = model.predict(expanded).reshape(
        n_rows, GRID_POINTS
    ).astype(bool)

    distance = np.abs(
        candidates - context["power_W"].to_numpy()[:, None]
    )

    distance[~is_normal] = np.inf

    best = distance.argmin(axis=1)

    found = np.isfinite(distance[np.arange(n_rows), best])

    return np.where(
        found,
        candidates[np.arange(n_rows), best],
        np.nan
    )


# =====================================================
# DiCE
# =====================================================

def build_dice_explainer(model, context_all, labels_all):

    data = context_all.copy()
    data["is_normal"] = (labels_all == "NORMAL").astype(int)

    data_interface = dice_ml.Data(
        dataframe=data,
        continuous_features=CONTEXT_COLUMNS,
        outcome_name="is_normal"
    )

    model_interface = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier"
    )

    return dice_ml.Dice(
        data_interface,
        model_interface,
        method="random"
    )


def dice_counterfactual_power(explainer, query):

    upper = float(power_upper_bound(query).iloc[0])

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        explanation = explainer.generate_counterfactuals(
            query,
            total_CFs=3,
            desired_class=1,
            features_to_vary=["power_W"],
            permitted_range={"power_W": [0.0, upper]},
            random_seed=42,
            verbose=False
        )

    frame = explanation.cf_examples_list[0].final_cfs_df

    if frame is None or frame.empty:
        return np.nan

    powers = frame["power_W"].astype(float)

    original = float(query["power_W"].iloc[0])

    # Closest counterfactual = smallest change
    return float(
        powers.iloc[(powers - original).abs().argmin()]
    )


# =====================================================
# RULE FALLBACK
# =====================================================

def rule_counterfactual_power(waste_type, context_row):

    """Used when neither DiCE nor the model search finds a
    power level the model calls NORMAL (e.g. the recent
    history alone keeps the reading unusual)."""

    if waste_type == "IE":
        return 0.0

    if waste_type == "OC":
        return min(
            context_row["power_W"],
            OVERCONSUMPTION_LIMIT_W - 0.01
        )

    # UP: back to the level before the sudden change
    return context_row["prev_mean_power"]


# =====================================================
# MESSAGES
# =====================================================

def build_message(waste_type, original, counterfactual, method):

    change = original - counterfactual

    if method == "rule":

        if waste_type == "IE":
            return (
                f"Equipment is drawing {original:.2f} W with "
                "nobody present. Switching it off (0 W) removes "
                "this idle waste."
            )

        if waste_type == "OC":
            return (
                f"Keeping power below {OVERCONSUMPTION_LIMIT_W:.1f} W "
                f"(-{change:.2f} W) and steady would end this "
                "overconsumption."
            )

        return (
            f"Power changed suddenly to {original:.2f} W from a "
            f"recent level of about {counterfactual:.2f} W. "
            "Check what was switched on or off at this moment."
        )

    change = original - counterfactual

    if counterfactual < 0.05:
        return (
            f"Switching the equipment off (from {original:.2f} W "
            "to about 0 W) would make this reading normal."
        )

    if change >= 0:
        return (
            f"Reducing power from {original:.2f} W to about "
            f"{counterfactual:.2f} W (-{change:.2f} W) would "
            "make this reading normal."
        )

    return (
        f"Power returning to about {counterfactual:.2f} W "
        f"(its recent level, +{-change:.2f} W) would make "
        "this reading normal."
    )


# =====================================================
# MAIN
# =====================================================

predictor = Predictor()

model = NormalityModel(predictor)

classified = pd.read_csv(CLASSIFIED_FILE)

anomalies = (
    classified[classified["waste_type"] != "NORMAL"]
    .tail(MAX_EXPLAIN_ROWS)
    .copy()
)

print("Waste events:", len(anomalies))

if anomalies.empty:

    pd.DataFrame(
        columns=[
            "reading_id", "timestamp", "waste_type",
            "original_power_W",
            "counterfactual_power_W", "power_reduction_W",
            "method", "if_present_class", "message"
        ]
    ).to_csv(COUNTERFACTUAL_FILE, index=False)

    print("Nothing to explain.")
    raise SystemExit


context = to_context(anomalies)

# -------------------------------------------------
# Model search for every event
# -------------------------------------------------

counterfactual_power = search_counterfactual_power(
    model, context
)

method = np.where(
    np.isnan(counterfactual_power),
    "none",
    "model_search"
)

# -------------------------------------------------
# DiCE for the most recent events
# -------------------------------------------------

if dice_ml is None:

    print("dice-ml not installed; using model search only.")

else:

    print(
        f"Running DiCE on the latest {MAX_DICE_ROWS} events..."
    )

    explainer = build_dice_explainer(
        model,
        to_context(classified),
        classified["waste_type"]
    )

    dice_success = 0

    for position in range(
        max(0, len(context) - MAX_DICE_ROWS),
        len(context)
    ):

        try:

            power = dice_counterfactual_power(
                explainer,
                context.iloc[[position]]
            )

        except Exception as error:

            print("  DiCE failed:", type(error).__name__)
            continue

        if not np.isnan(power):

            counterfactual_power[position] = power
            method[position] = "dice"
            dice_success += 1

    print(
        f"DiCE found counterfactuals for "
        f"{dice_success} events."
    )

# -------------------------------------------------
# Rule fallback where nothing was found
# -------------------------------------------------

for position in np.flatnonzero(np.isnan(counterfactual_power)):

    counterfactual_power[position] = rule_counterfactual_power(
        anomalies["waste_type"].iloc[position],
        context.iloc[position]
    )

    method[position] = "rule"

# -------------------------------------------------
# "What if someone were present?" (useful for IE)
# -------------------------------------------------

occupied = context.copy()
occupied["presence_binary"] = 1

if_present_class = predictor.classes[
    model.class_probabilities(occupied).argmax(axis=1)
]


# =====================================================
# SAVE
# =====================================================

original_power = anomalies["power_W"].to_numpy()

results = pd.DataFrame({
    "reading_id": anomalies["reading_id"].to_numpy(),
    "timestamp": anomalies["timestamp"].to_numpy(),
    "waste_type": anomalies["waste_type"].to_numpy(),
    "original_power_W": original_power.round(3),
    "counterfactual_power_W": np.round(counterfactual_power, 3),
    "power_reduction_W": np.round(
        original_power - counterfactual_power, 3
    ),
    "method": method,
    "if_present_class": if_present_class,
    "message": [
        build_message(waste, original, cf, how)
        for waste, original, cf, how in zip(
            anomalies["waste_type"],
            original_power,
            counterfactual_power,
            method
        )
    ]
})

results.to_csv(COUNTERFACTUAL_FILE, index=False)


# =====================================================
# SUMMARY
# =====================================================

print()
print("Method used:")
print(results["method"].value_counts().to_string())

print()
print("Median power reduction by class (W):")
print(
    results.groupby("waste_type")["power_reduction_W"]
    .median()
    .round(3)
    .to_string()
)

print()
print(results.tail(5).to_string(index=False))

print()
print("Output:", COUNTERFACTUAL_FILE)

print()
print("======================================")
print("DiCE COMPLETE")
print("======================================")
