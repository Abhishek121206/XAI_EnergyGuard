import json

import pandas as pd
import pytest

import app as api


@pytest.fixture
def client(tmp_path, monkeypatch):

    classified = pd.DataFrame({
        "reading_id": ["1_0", "2_0", "3_0", "4_0"],
        "timestamp": [1, 2, 3, 4],
        "power_W": [0.5, 9.0, 2.0, float("nan")],
        "waste_type": ["NORMAL", "IE", "OC", "NORMAL"],
        "is_outlier": [0, 1, 0, 0]
    })

    explanations = pd.DataFrame({
        "reading_id": ["2_0", "3_0"],
        "waste_type": ["IE", "OC"],
        "facility_manager": ["a", "b"]
    })

    classified_file = tmp_path / "classified.csv"
    explanations_file = tmp_path / "explanations.csv"
    metrics_file = tmp_path / "metrics.json"

    classified.to_csv(classified_file, index=False)
    explanations.to_csv(explanations_file, index=False)
    metrics_file.write_text(json.dumps({"trained_at": "v1"}))

    monkeypatch.setattr(api, "CLASSIFIED_FILE", classified_file)
    monkeypatch.setattr(api, "EXPLANATIONS_FILE", explanations_file)
    monkeypatch.setattr(api, "METRICS_FILE", metrics_file)
    monkeypatch.setattr(
        api, "LATEST_PREDICTION_FILE", tmp_path / "missing.json"
    )

    return api.app.test_client()


def test_health(client):

    body = client.get("/api/health").get_json()

    assert body["status"] == "healthy"
    assert body["model_trained"] is True


def test_data_limit_and_nan(client):

    response = client.get("/api/data?limit=2")

    # NaN must become null, not invalid JSON
    body = json.loads(response.data)

    assert [row["timestamp"] for row in body] == [3, 4]
    assert body[1]["power_W"] is None


def test_anomalies_are_non_normal(client):

    body = client.get("/api/anomalies").get_json()

    assert [row["waste_type"] for row in body] == ["IE", "OC"]


def test_latest_anomaly(client):

    body = client.get("/api/anomalies/latest").get_json()

    assert body["waste_type"] == "OC"


def test_summary(client):

    body = client.get("/api/summary").get_json()

    assert body["total_readings"] == 4
    assert body["anomaly_count"] == 2
    assert body["class_counts"] == {
        "NORMAL": 2, "IE": 1, "OC": 1, "UP": 0
    }
    assert body["outlier_count"] == 1
    assert body["model_version"] == "v1"


def test_metrics_and_missing_live(client):

    assert client.get("/api/model/metrics").get_json() == {
        "trained_at": "v1"
    }

    assert client.get("/api/live").status_code == 404


def test_cors_only_dashboard(client):

    allowed = client.get(
        "/api/health",
        headers={"Origin": "http://127.0.0.1:5500"}
    )
    blocked = client.get(
        "/api/health",
        headers={"Origin": "http://evil.example"}
    )

    assert "Access-Control-Allow-Origin" in allowed.headers
    assert "Access-Control-Allow-Origin" not in blocked.headers
