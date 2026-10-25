import pandas as pd

from ml_config import (
    CLASS_NAMES,
    CLASSIFIED_FILE,
    COUNTERFACTUAL_FILE,
    EXPLANATIONS_FILE,
    MAX_EXPLAIN_ROWS,
    SHAP_FILE,
    SPIKE_MIN_W,
    SPIKE_STD_MULTIPLIER
)


TOP_N = 3


# =====================================================
# FEATURE -> PLAIN LANGUAGE
# =====================================================

def describe_feature(feature, value):

    """Turn one SHAP feature and its value into a reason."""

    if feature == "presence_binary":
        return (
            "the room was occupied"
            if value >= 0.5
            else "nobody was present in the room"
        )

    if feature == "power_W":
        return f"power draw was {value:.2f} W"

    if feature == "current_A":
        return f"current was {value:.3f} A"

    if feature == "rolling_mean_power":
        return f"average power over the last few seconds was {value:.2f} W"

    if feature == "rolling_std_power":
        return (
            f"power was fluctuating (spread {value:.2f} W)"
            if value >= 0.3
            else "power was steady"
        )

    if feature == "prev_rolling_std_power":
        return (
            f"power had been fluctuating beforehand (spread {value:.2f} W)"
            if value >= 0.3
            else "power had been steady beforehand"
        )

    if feature == "power_jump":
        if abs(value) < SPIKE_MIN_W:
            return "power was close to its recent level"
        direction = "rose" if value >= 0 else "fell"
        return (
            f"power {direction} by {abs(value):.2f} W compared "
            "with the previous seconds"
        )

    if feature == "power_jump_score":
        if value < SPIKE_STD_MULTIPLIER:
            return "there was no sudden change in power"
        return (
            f"power changed by {value:.1f}x its usual fluctuation"
        )

    if feature == "deviation_from_baseline":
        side = "above" if value >= 0 else "below"
        return (
            f"power was {abs(value):.2f} W {side} the normal "
            "baseline for this hour"
        )

    return f"{feature} was {value}"


def shap_reasons(row):

    """Reasons that pushed the model TOWARDS the predicted
    class (positive SHAP), strongest first."""

    reasons = []

    for rank in range(1, TOP_N + 1):

        feature = row.get(f"top{rank}_feature")
        value = row.get(f"top{rank}_value")
        shap_value = row.get(f"top{rank}_shap")

        if pd.isna(feature) or pd.isna(value):
            continue

        if shap_value is not None and shap_value <= 0 and reasons:
            continue

        reasons.append(describe_feature(feature, float(value)))

    return reasons


def join_reasons(reasons):

    if not reasons:
        return "the overall pattern of recent readings"

    if len(reasons) == 1:
        return reasons[0]

    return ", ".join(reasons[:-1]) + " and " + reasons[-1]


def counterfactual_text(row):

    message = row.get("message")

    if isinstance(message, str) and message:
        return message

    return ""


# =====================================================
# AUDIENCE MESSAGES
# =====================================================

def facility_manager_message(row):

    label = CLASS_NAMES[row["waste_type"]].upper()

    text = (
        f"{label} detected "
        f"({row['waste_confidence']:.0%} model confidence) "
        f"at {row['power_W']:.2f} W. "
        f"Why: {join_reasons(shap_reasons(row))}."
    )

    action = counterfactual_text(row)

    if action:
        text += f" What would make it normal: {action}"

    return text


def student_message(row):

    power = row["power_W"]
    waste_type = row["waste_type"]

    if waste_type == "IE":
        return (
            f"Equipment here is using about {power:.2f} W while "
            "nobody is in the room. Please switch things off "
            "when you leave."
        )

    if waste_type == "OC":
        return (
            f"Energy use is high (about {power:.2f} W) while the "
            "room is in use. Please switch off anything you are "
            "not using."
        )

    if row.get("power_jump", 0) >= 0:
        return (
            f"Power use jumped suddenly to about {power:.2f} W. "
            "If you just switched something on, check it is needed."
        )

    return (
        f"Power use dropped suddenly to about {power:.2f} W. "
        "If equipment stopped unexpectedly, please report it."
    )


def administrator_message(row):

    label = CLASS_NAMES[row["waste_type"]]

    reduction = row.get("power_reduction_W")

    text = f"{label} event. Observed power: {row['power_W']:.2f} W."

    if pd.notna(reduction) and reduction > 0:
        text += (
            f" Avoidable load about {reduction:.2f} W "
            f"(~{reduction:.1f} Wh per hour if it continues)."
        )

    if row["waste_type"] == "IE":
        text += " Review shutdown procedures for unoccupied periods."

    elif row["waste_type"] == "OC":
        text += " Review equipment usage during occupied hours."

    else:
        text += " Investigate the equipment switched at this time."

    return text


# =====================================================
# LOAD + MERGE
# =====================================================

print("======================================")
print("XAI-EnergyGuard NLG")
print("======================================")

classified_df = pd.read_csv(CLASSIFIED_FILE)
shap_df = pd.read_csv(SHAP_FILE)
dice_df = pd.read_csv(COUNTERFACTUAL_FILE)

print("Classified rows     :", len(classified_df))
print("SHAP rows           :", len(shap_df))
print("Counterfactual rows :", len(dice_df))

anomalies = (
    classified_df[classified_df["waste_type"] != "NORMAL"]
    .tail(MAX_EXPLAIN_ROWS)
    [[
        "reading_id", "timestamp", "power_W", "power_jump",
        "waste_type", "waste_confidence"
    ]]
)

shap_columns = ["reading_id"] + [
    f"top{rank}_{part}"
    for rank in range(1, TOP_N + 1)
    for part in ("feature", "value", "shap")
]

anomalies = anomalies.merge(
    shap_df[shap_columns],
    on="reading_id",
    how="left"
).merge(
    dice_df[[
        "reading_id",
        "counterfactual_power_W",
        "power_reduction_W",
        "method",
        "message"
    ]],
    on="reading_id",
    how="left"
)


# =====================================================
# GENERATE
# =====================================================

results = pd.DataFrame({
    "reading_id": anomalies["reading_id"],
    "timestamp": anomalies["timestamp"],
    "waste_type": anomalies["waste_type"],
    "waste_confidence": anomalies["waste_confidence"],
    "power_W": anomalies["power_W"],
    "facility_manager": [
        facility_manager_message(row)
        for row in anomalies.to_dict("records")
    ],
    "student": [
        student_message(row)
        for row in anomalies.to_dict("records")
    ],
    "administrator": [
        administrator_message(row)
        for row in anomalies.to_dict("records")
    ],
    "counterfactual": anomalies["message"],
    "counterfactual_method": anomalies["method"]
})

results.to_csv(EXPLANATIONS_FILE, index=False)


# =====================================================
# DISPLAY
# =====================================================

print()
print("Explanations generated:", len(results))

print()
print("Example per class:")

for waste_type, group in results.groupby("waste_type"):

    example = group.iloc[-1]

    print()
    print(f"[{waste_type}]")
    print("  Facility manager:", example["facility_manager"])
    print("  Student         :", example["student"])
    print("  Administrator   :", example["administrator"])

print()
print("Output:", EXPLANATIONS_FILE)

print()
print("======================================")
print("NLG COMPLETE")
print("======================================")
