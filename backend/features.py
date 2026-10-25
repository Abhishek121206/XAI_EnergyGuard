import numpy as np
import pandas as pd

from ml_config import (
    CURRENT_NOISE_LIMIT,
    OVERCONSUMPTION_LIMIT_W,
    ROLLING_WINDOW,
    SESSION_GAP_SECONDS,
    SPIKE_MIN_W,
    SPIKE_STD_MULTIPLIER,
    STATUS_TO_CLASS
)


# =====================================================
# OUTPUT COLUMNS
# =====================================================

FEATURE_OUTPUT_COLUMNS = [
    "reading_id",
    "timestamp",
    "circuit_id",
    "status",
    "distance_cm",
    "power_W",
    "current_A",
    "voltage_V",
    "temperature_C",
    "humidity_percent",
    "presence_binary",
    "hour",
    "day_of_week",
    "day_type",
    "session_id",
    "rolling_mean_power",
    "rolling_std_power",
    "prev_rolling_std_power",
    "power_jump",
    "power_jump_score",
    "baseline_power",
    "deviation_from_baseline"
]


# =====================================================
# BASELINE
# =====================================================

def compute_baseline(features_df):

    """Hourly median power of normal-use rows.

    Returns {"by_hour": {hour: watts}, "global": watts}.
    Falls back to all rows if no normal rows exist.
    """

    rows = features_df

    if "status" in features_df.columns:

        normal = features_df[
            features_df["status"]
            .map(STATUS_TO_CLASS)
            .eq("NORMAL")
        ]

        if len(normal) > 0:
            rows = normal

    by_hour = (
        rows.groupby("hour")["power_W"]
        .median()
    )

    return {
        "by_hour": {
            int(hour): float(value)
            for hour, value in by_hour.items()
        },
        "global": float(rows["power_W"].median())
    }


def apply_baseline(features_df, baseline):

    df = features_df

    df["baseline_power"] = (
        df["hour"]
        .map(baseline["by_hour"])
        .fillna(baseline["global"])
        .astype(float)
    )

    df["deviation_from_baseline"] = (
        df["power_W"] - df["baseline_power"]
    )

    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_features(raw_df, baseline=None):

    """Turn raw sensor readings into model features.

    baseline: table from compute_baseline(). When None it is
    computed from raw_df itself (only use that for training).
    """

    df = raw_df.copy()

    for column in ["circuit_id", "status", "distance_cm"]:

        if column not in df.columns:
            df[column] = np.nan

    # -------------------------------------------------
    # CLEAN + SORT
    # -------------------------------------------------

    df["presence_binary"] = df["presence"].map({
        "PRESENT": 1,
        "ABSENT": 0
    })

    df = df.dropna(
        subset=[
            "timestamp",
            "power_W",
            "current_A",
            "presence_binary"
        ]
    )

    df = df.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(drop=True)

    df["presence_binary"] = (
        df["presence_binary"].astype(int)
    )

    # Unique per reading (the ESP32 sends ~2 per second)
    df["reading_id"] = (
        df["timestamp"].astype("int64").astype(str)
        + "_"
        + df.groupby("timestamp").cumcount().astype(str)
    )

    # -------------------------------------------------
    # TIME FEATURES
    # -------------------------------------------------

    datetime = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    df["hour"] = datetime.dt.hour
    df["day_of_week"] = datetime.dt.dayofweek
    df["day_type"] = (df["day_of_week"] >= 5).astype(int)

    # -------------------------------------------------
    # SESSIONS
    # -------------------------------------------------

    df["session_id"] = (
        df["timestamp"].diff() > SESSION_GAP_SECONDS
    ).cumsum()

    # -------------------------------------------------
    # ROLLING FEATURES (per session)
    # -------------------------------------------------

    sessions = df.groupby("session_id")["power_W"]

    df["rolling_mean_power"] = sessions.transform(
        lambda s: s.rolling(
            ROLLING_WINDOW,
            min_periods=1
        ).mean()
    )

    df["rolling_std_power"] = sessions.transform(
        lambda s: s.rolling(
            ROLLING_WINDOW,
            min_periods=2
        ).std()
    ).fillna(0.0)

    # Statistics of the window BEFORE this reading

    previous_mean = (
        df.groupby("session_id")["rolling_mean_power"]
        .shift(1)
    )

    df["prev_rolling_std_power"] = (
        df.groupby("session_id")["rolling_std_power"]
        .shift(1)
        .fillna(0.0)
    )

    df["power_jump"] = (
        df["power_W"]
        - previous_mean.fillna(df["power_W"])
    )

    # How many "previous standard deviations" the jump is
    df["power_jump_score"] = (
        df["power_jump"].abs()
        / df["prev_rolling_std_power"].clip(
            lower=SPIKE_MIN_W / SPIKE_STD_MULTIPLIER
        )
    )

    # -------------------------------------------------
    # BASELINE
    # -------------------------------------------------

    if baseline is None:
        baseline = compute_baseline(df)

    df = apply_baseline(df, baseline)

    return df[FEATURE_OUTPUT_COLUMNS].copy()


# =====================================================
# LABELS
# =====================================================

def rule_class(features_df):

    """Firmware rule recomputed from the readings."""

    absent_with_load = (
        (features_df["presence_binary"] == 0)
        & (features_df["current_A"] > CURRENT_NOISE_LIMIT)
    )

    over_limit = (
        features_df["power_W"] > OVERCONSUMPTION_LIMIT_W
    )

    labels = pd.Series(
        "NORMAL",
        index=features_df.index
    )

    labels[over_limit] = "OC"
    labels[absent_with_load] = "IE"

    return labels


def spike_mask(features_df):

    threshold = np.maximum(
        SPIKE_STD_MULTIPLIER
        * features_df["prev_rolling_std_power"],
        SPIKE_MIN_W
    )

    return (
        (features_df["presence_binary"] == 1)
        & (features_df["power_jump"].abs() > threshold)
    )


def make_labels(features_df):

    """Training labels.

    IE / OC / NORMAL come from the firmware status logged by
    the ESP32 (recomputed from the rule when it is missing).
    UP marks occupied readings with a sudden power change.
    """

    labels = (
        features_df["status"]
        .map(STATUS_TO_CLASS)
    )

    labels = labels.fillna(
        rule_class(features_df)
    )

    labels[
        spike_mask(features_df)
        & (labels != "IE")
    ] = "UP"

    return labels
