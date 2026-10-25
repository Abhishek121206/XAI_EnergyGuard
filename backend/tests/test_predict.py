import pytest

from conftest import reading
from ml_config import CLASSES, MODEL_BUNDLE_FILE

pytestmark = pytest.mark.skipif(
    not MODEL_BUNDLE_FILE.exists(),
    reason="no trained model (run python train.py)"
)


@pytest.fixture(scope="module")
def predictor():

    from predict import Predictor

    return Predictor()


def test_predict_raw_schema(predictor, raw_df):

    result = predictor.predict_raw(raw_df)

    assert len(result) == len(raw_df)
    assert set(result["waste_type"]) <= set(CLASSES)
    assert result["waste_confidence"].between(0, 1).all()

    for column in (
        "anomaly", "anomaly_score", "is_outlier",
        *(f"prob_{label}" for label in CLASSES)
    ):
        assert column in result.columns


def test_obvious_cases(predictor, raw_df):

    result = predictor.predict_raw(raw_df)

    assert (result["waste_type"].iloc[2:15] == "NORMAL").all()
    assert (result["waste_type"].iloc[17:] == "IE").all()


def test_live_matches_batch(predictor, raw_df):

    from predict import Predictor

    live = Predictor()

    live_results = [
        live.predict_reading(row)
        for row in raw_df.to_dict("records")
    ]

    batch = predictor.predict_raw(raw_df)

    assert [r["waste_type"] for r in live_results] == list(
        batch["waste_type"]
    )


def test_live_reading_is_json_safe(predictor):

    from predict import Predictor

    live = Predictor()

    result = live.predict_reading(
        reading(1_000, 0.8, "ABSENT", None)
    )

    assert result["waste_type"] in CLASSES
    assert result["model_version"] == predictor.version
