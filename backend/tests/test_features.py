import pandas as pd

from features import (
    build_features,
    compute_baseline,
    make_labels,
    rule_class
)
from features import FEATURE_OUTPUT_COLUMNS
from ml_config import MODEL_FEATURES


def test_output_schema(raw_df):

    features = build_features(raw_df)

    assert list(features.columns) == FEATURE_OUTPUT_COLUMNS
    assert set(MODEL_FEATURES) <= set(features.columns)
    assert len(features) == len(raw_df)
    assert features["reading_id"].is_unique


def test_duplicate_timestamps_get_unique_ids(raw_df):

    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]])

    features = build_features(duplicated)

    assert features["reading_id"].is_unique


def test_rolling_windows_do_not_cross_sessions(raw_df):

    features = build_features(raw_df)

    first_of_second_session = features[
        features["timestamp"] == 5_000
    ].iloc[0]

    assert features["session_id"].nunique() == 2

    # No history from the first session leaks in
    assert first_of_second_session["rolling_mean_power"] == 9.0
    assert first_of_second_session["power_jump"] == 0.0


def test_given_baseline_is_used(raw_df):

    baseline = {"by_hour": {}, "global": 2.0}

    features = build_features(raw_df, baseline)

    assert (features["baseline_power"] == 2.0).all()
    assert (
        features["deviation_from_baseline"]
        == features["power_W"] - 2.0
    ).all()


def test_baseline_uses_normal_rows():

    features = pd.DataFrame({
        "hour": [10, 10, 10],
        "power_W": [1.0, 1.0, 50.0],
        "status": ["NORMAL", "NORMAL", "OVERCONSUMPTION"]
    })

    baseline = compute_baseline(features)

    assert baseline["by_hour"][10] == 1.0


def test_labels(raw_df):

    features = build_features(raw_df)

    labels = make_labels(features)

    # Steady occupied use is normal
    assert (labels.iloc[:15] == "NORMAL").all()

    # Sudden jump from 0.8 W to 6 W while occupied
    assert labels.iloc[15] == "UP"

    # Empty room with load on
    assert (labels.iloc[16:] == "IE").all()


def test_missing_status_falls_back_to_rule(raw_df):

    raw_df["status"] = None

    features = build_features(raw_df)

    labels = make_labels(features)

    assert (labels.iloc[16:] == "IE").all()
    assert (labels.iloc[:15] == "NORMAL").all()


def test_rule_class_matches_firmware():

    features = pd.DataFrame({
        "presence_binary": [0, 0, 1, 1],
        "current_A": [0.0, 0.2, 0.2, 0.4],
        "power_W": [0.0, 1.0, 1.0, 2.0]
    })

    assert list(rule_class(features)) == [
        "NORMAL", "IE", "NORMAL", "OC"
    ]
