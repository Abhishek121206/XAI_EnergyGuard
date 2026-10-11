import pandas as pd
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

CLASSIFIED_FILE = BASE_DIR / "data" / "c1_classified.csv"
SHAP_FILE = BASE_DIR / "data" / "c1_shap.csv"
DICE_FILE = BASE_DIR / "data" / "c1_counterfactuals.csv"

OUTPUT_FILE = BASE_DIR / "data" / "c1_explanations.csv"


# =====================================================
# FEATURE NAME TRANSLATION
# =====================================================

FEATURE_NAMES = {

    "power_W":
        "power consumption",

    "current_A":
        "current",

    "voltage_V":
        "voltage",

    "temperature_C":
        "temperature",

    "humidity_percent":
        "humidity",

    "presence_binary":
        "room presence",

    "hour":
        "time of day",

    "day_type":
        "weekday/weekend pattern",

    "rolling_mean_power":
        "recent average power",

    "rolling_std_power":
        "power variation",

    "baseline_power":
        "normal baseline power",

    "deviation_from_baseline":
        "deviation from normal power"
}


# =====================================================
# LOAD FILES
# =====================================================

print("======================================")
print("XAI-EnergyGuard NLG")
print("======================================")

classified_df = pd.read_csv(
    CLASSIFIED_FILE
)

shap_df = pd.read_csv(
    SHAP_FILE
)

dice_df = pd.read_csv(
    DICE_FILE
)


print("Classified rows:", len(classified_df))
print("SHAP rows      :", len(shap_df))
print("DiCE rows      :", len(dice_df))


# =====================================================
# SELECT ANOMALIES
# =====================================================

anomalies = classified_df[
    classified_df["anomaly"] == "ANOMALY"
].copy()


if len(anomalies) == 0:

    print()
    print("No anomalies found.")
    raise SystemExit


# =====================================================
# MERGE SHAP DATA
# =====================================================


shap_columns = [
    "timestamp",
    "top1_feature",
    "top1_shap",
    "top2_feature",
    "top2_shap",
    "top3_feature",
    "top3_shap"
]

anomalies = anomalies.merge(
    shap_df[shap_columns],
    on="timestamp",
    how="left"
)

# =====================================================
# MERGE DiCE DATA
# =====================================================

anomalies = anomalies.merge(
    dice_df[
        [
            "timestamp",
            "original_power_W",
            "counterfactual_power_W",
            "power_reduction_W",
            "message"
        ]
    ],
    on="timestamp",
    how="left",
    suffixes=("", "_dice")
)


# =====================================================
# SHAP DESCRIPTION
# =====================================================

def get_feature_name(feature):

    if pd.isna(feature):

        return "unknown factor"

    return FEATURE_NAMES.get(
        feature,
        str(feature)
    )


def build_shap_text(row):

    features = [

        get_feature_name(
            row.get("top1_feature")
        ),

        get_feature_name(
            row.get("top2_feature")
        ),

        get_feature_name(
            row.get("top3_feature")
        )
    ]

    return (
        f"{features[0]}, "
        f"{features[1]}, and "
        f"{features[2]}"
    )


# =====================================================
# FACILITY MANAGER EXPLANATION
# =====================================================

def facility_manager_message(row):

    waste_type = row["waste_type"]

    power = row["power_W"]

    shap_text = build_shap_text(row)

    dice_message = row.get(
        "message",
        None
    )


    if waste_type == "IE":

        message = (
            "IDLE WASTE detected. "
            f"Current power draw is {power:.3f} W "
            "while no room presence was detected. "
            f"Main contributing factors: {shap_text}."
        )

    elif waste_type == "OC":

        message = (
            "OVERCONSUMPTION detected. "
            f"Current power draw is {power:.3f} W. "
            f"Main contributing factors: {shap_text}."
        )

    else:

        message = (
            "UNUSUAL PATTERN detected. "
            f"Current power draw is {power:.3f} W. "
            f"Main contributing factors: {shap_text}. "
            "Manual investigation is recommended."
        )


    if (
        isinstance(dice_message, str)
        and
        dice_message.startswith(
            "Reducing power"
        )
    ):

        message += " " + dice_message


    return message


# =====================================================
# STUDENT EXPLANATION
# =====================================================

def student_message(row):

    waste_type = row["waste_type"]

    power = row["power_W"]


    if waste_type == "IE":

        return (
            "Energy waste detected in this area. "
            f"The system detected about {power:.3f} W "
            "of power use while no presence was detected. "
            "Please make sure equipment is switched off "
            "when the room is not being used."
        )


    elif waste_type == "OC":

        return (
            "Higher-than-usual energy use was detected "
            f"at about {power:.3f} W while the area "
            "was occupied. Please check whether equipment "
            "or cooling is being used unnecessarily."
        )


    else:

        return (
            "The system detected an unusual energy pattern "
            f"of about {power:.3f} W. "
            "Please check the area if the unusual usage "
            "continues."
        )


# =====================================================
# ADMINISTRATOR EXPLANATION
# =====================================================

def administrator_message(row):

    waste_type = row["waste_type"]

    power = row["power_W"]


    if waste_type == "IE":

        return (
            "Idle-equipment energy event detected. "
            f"Observed power: {power:.3f} W. "
            "Review after-hours shutdown procedures."
        )


    elif waste_type == "OC":

        return (
            "Overconsumption event detected. "
            f"Observed power: {power:.3f} W. "
            "Review equipment usage and operating conditions."
        )


    else:

        return (
            "Unusual energy pattern detected. "
            f"Observed power: {power:.3f} W. "
            "Manual investigation is recommended."
        )


# =====================================================
# GENERATE EXPLANATIONS
# =====================================================

results = []


for _, row in anomalies.iterrows():

    results.append({

        "timestamp":
            row["timestamp"],

        "waste_type":
            row["waste_type"],

        "waste_confidence":
            row["waste_confidence"],

        "facility_manager":
            facility_manager_message(row),

        "student":
            student_message(row),

        "administrator":
            administrator_message(row)
    })


# =====================================================
# SAVE
# =====================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# DISPLAY
# =====================================================

print()
print("======================================")
print("GENERATED EXPLANATIONS")
print("======================================")

print(
    results_df.to_string(
        index=False
    )
)

print()
print("Output:", OUTPUT_FILE)

print()
print("======================================")
print("NLG COMPLETE")
print("======================================")